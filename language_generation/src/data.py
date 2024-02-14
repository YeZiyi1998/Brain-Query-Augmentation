import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import gc
import json
import copy
from sklearn.preprocessing import StandardScaler
import sys
import os

class Splited_FMRI_dataset(Dataset):
    def __init__(self,inputs,most_epoch=-1, args = None,fpm=None,q_scores=None,tokenizer=None):
        self.device = torch.device(f"cuda:{args['cuda']}") if args['cuda'] != -1 else torch.device(f"cpu")
        self.inputs = inputs
        self.most_epoch = most_epoch
        self.args = args
        self.mode = 'document' if 'Pereira' in self.args['task_name'] else 'passage'
        self.fpm = fpm
        self.q_scores = q_scores
        self.tokenizer = tokenizer
        if 'token' in self.args['mode']:
            self.get_token_possibility()
    def __len__(self):
        if self.most_epoch > -1:
            return min(self.most_epoch, len(self.inputs))
        return len(self.inputs)
    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample 

class FMRI_dataset():
    def __init__(self, input_dataset, args, tokenizer,decoding_model=None):
        self.decoding_model = decoding_model
        self.args = args
        self.add_special_tokens=False
        self.shuffle_times = args['shuffle_times']
        self.tokenizer = tokenizer
        self.data_path = f'../../../common_data/{args["task_name"]}/'
        os.makedirs(self.data_path, exist_ok=True)
        self.prefix = args["common_data_prefix"] + str(args["split_num"]) + '' if args['data_spliting'] == 'random' else args['data_spliting']
        if os.path.exists(self.data_path+f'{self.prefix}all_input_data.pkl') and args['reload'] == False:
            self.inputs = pickle.load(open(self.data_path+f'{self.prefix}all_input_data.pkl', 'rb'))  
        else:
            from data_old import FMRI_dataset_old  
            old_dataset = FMRI_dataset_old(input_dataset, args, tokenizer,decoding_model, prefix= self.prefix )
            pickle.dump(old_dataset.inputs, open(self.data_path+f'{self.prefix}all_input_data.pkl', 'wb'))
            self.inputs = old_dataset.inputs
        self.inputs = [{key: value for key, value in item.items() if value is not None} for item in self.inputs]
        self.pack_data_from_input(args)
    
    def pack_data_from_input(self, args, ):
        self.train = []
        self.test = []
        self.valid = []
        self.is_shuffled = False
        test_ids = args['test_trail_ids']
        valid_ids = args['valid_trail_ids']
        for idx,item in enumerate(self.inputs):
            item_pack = False
            if item['trail_id'] > test_ids[0] and item['trail_id'] <= test_ids[1]:
                self.test.append(item)
                item_pack = True
            if item['trail_id'] > valid_ids[0] and item['trail_id'] <= valid_ids[1]:
                self.valid.append(item)
                item_pack = True
            if item_pack == False:
                self.train.append(item)
        if args['input_method'] == 'permutated':
            tmp_additional_bs = copy.deepcopy([self.test[(idx+int(len(self.test)/2))%len(self.test)]['additional_bs'] for idx in range(len(self.test))])
            random.shuffle(tmp_additional_bs)
            for idx,item in enumerate(self.test):
                self.test[idx]['additional_bs'] = tmp_additional_bs[idx]
        if args['data_size'] != -1:
            random.shuffle(self.train)
            self.train = self.train[:args['data_size']]
        
        # get hard negatives
        # q_scores = pickle.load(open(f'../../../ict/results/{args["task_name"]}/repllama_rank_q_scores.pkl','rb'))
        q_scores = None
        fpm = pickle.load(open(f'../../../common_data/{args["task_name"]}/{self.prefix}fdm.pkl','rb')) if 'Pereira' in self.args['task_name'] else pickle.load(open(f'../../../common_data/{args["task_name"]}/{self.prefix}fpm.pkl','rb'))
        self.train_dataset = Splited_FMRI_dataset(self.train, args = args,q_scores=q_scores,fpm=fpm,tokenizer = self.tokenizer)
        self.valid_dataset = Splited_FMRI_dataset(self.valid, args = args,q_scores=q_scores,fpm=fpm,tokenizer = self.tokenizer)
        self.test_dataset = Splited_FMRI_dataset(self.test, args = args,q_scores=q_scores,fpm=fpm,tokenizer = self.tokenizer)
        self.all_dataset = Splited_FMRI_dataset(self.inputs, args = args,q_scores=q_scores,fpm=fpm,tokenizer = self.tokenizer)
    
    # expanding may be difficult because we also need to change the all_input_data file
    def expand(self, test_dataset):
        if os.path.exists(self.data_path+f'{self.prefix}all_input_data.expand.pkl') and self.args['reload'] == False:  
            expand_dataset = []
        for item in test_dataset:
            content_true_length = torch.sum(item['content_true_mask'])
            content_pre_length = torch.sum(item['content_prev_mask'])
            for j in range(content_true_length - 1):
                if content_pre_length < len(item['content_prev']):
                    item['content_prev'][content_pre_length] = item['content_true'][0]
                    item['content_prev_mask'][content_pre_length] = 1
                    content_pre_length += 1
                else:
                    item['content_prev'] = torch.cat([item['content_prev'][:-1], item['content_true'][0].unsqueeze(0)])
                item['content_true'] = torch.cat([item['content_true'][1:], item['content_true'][-1:]])
                item['content_true_mask'][content_true_length-1] = 0
                content_true_length -= 1
                expand_dataset.append(copy.deepcopy(item))
        return expand_dataset
    