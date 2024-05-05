import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
import json
from model_config import model_name2path, model2hidden
from model_utils import Prompt_model
import wandb 
import torch.optim.lr_scheduler as lr_scheduler
from optimizer import Adam16
import random
from peft import PeftModel, PeftConfig
import gc 
import numpy as np

class Decoding_model:
    def put_data_into_cuda(self, content_prev,additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, ):
        content_prev, content_prev_sep, content_true, content_prev_mask, content_true_mask = content_prev.to(self.device), content_prev_sep.to(self.device), content_true.to(self.device), content_prev_mask.to(self.device), content_true_mask.to(self.device)
        if type(additional_bs) == list:
            for k in range(len(additional_bs)):
                additional_bs[k] = additional_bs[k].to(self.device)
        else:
            additional_bs = additional_bs.to(self.device)
        if self.args['model_name'] in ['llama-7b','st']:
            additional_bs_mask = torch.ones([additional_bs.shape[0], additional_bs.shape[1]+2+1]).to(self.device)
        else:
            additional_bs_mask = torch.ones([additional_bs.shape[0], additional_bs.shape[1]+2]).to(self.device)
        if self.args['model_name'] in ['llama-7b',]:
            additional_bs = additional_bs.half()
        return content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask
    
    def __init__(self, args):
        # load model
        self.device = torch.device(f"cuda:{args['cuda']}") if args['cuda'] != -1 else torch.device(f"cpu")
        self.args = args
        self.inv_temperature = 20 if args['model_name'] in ['llama-7b',] else 1
        if args['model_name'] in ['llama-7b',]:
            if args['model_name'] in model_name2path.keys():
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name2path[args['model_name']])
                if self.args['mode'] in ['generation_train','evaluate','token_train','token_evaluate','entrophy']:
                    self.model = LlamaForCausalLM.from_pretrained(model_name2path[args['model_name']]).to(self.device)
                else:
                    self.model = AutoModel.from_pretrained(model_name2path[args['model_name']]).to(self.device)
            else:
                self.tokenizer = LlamaTokenizer.from_pretrained(args['model_name'])
                self.model = LlamaForCausalLM.from_pretrained(args['model_name']).to(self.device)
            self.model.half()
        elif 'gpt' in args['model_name']:
            if args['model_name'] in model_name2path.keys():
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name2path[args['model_name']])
                self.model = GPT2LMHeadModel.from_pretrained(model_name2path[args['model_name']]).to(self.device)
            else:
                self.tokenizer = GPT2Tokenizer.from_pretrained(args['model_name'])
                self.model = GPT2LMHeadModel.from_pretrained(args['model_name']).to(self.device)
        elif 'st' in args['model_name']:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name2path[args['model_name']])
            self.model = AutoModel.from_pretrained(model_name2path[args['model_name']]).to(self.device)
        # add special token <brain/> and </brain>
        # if args['model_name'] not in ['llama-7b', 'vicuna-7b',]:
        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.eos_token = self.tokenizer.mask_token
            self.tokenizer.pad_token = self.tokenizer.mask_token
        
        if len(args['roi_selected']) > 0:
            self.new_tokens = []
            for k in range(len(args['roi_selected'])):
                self.new_tokens += ([f"<roi{k}/>", f"</roi{k}>"])
            self.tokenizer.add_tokens(self.new_tokens )
        self.new_tokens = ["<brain/>", "</brain>"]
        self.tokenizer.add_tokens(self.new_tokens)

        if args['model_name'] in ['llama-7b', 'vicuna-7b','llama-7b-old','st']:
            self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        else:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if args['enable_grad']==False:
            self.freeze_model()
        args['word_embed_size'] = model2hidden[args['model_name']]

        if args['enable_grad']==False:
            for new_token in self.new_tokens:
                # 发现bug
                # new_token_id = self.tokenizer.convert_tokens_to_ids(f"[{new_token}]")
                new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
                if 'gpt2' in self.args['model_name']:
                    self.model.transformer.wte.weight[new_token_id].requires_grad = True
                elif 'llama' in self.args['model_name']:
                    if self.args['mode'] in ['generation_train','evaluate','token_train','token_evaluate','entrophy']:
                        self.model.model.embed_tokens.weight[new_token_id].requires_grad = True
                    else:
                        self.model.embed_tokens.weight[new_token_id].requires_grad = True
                elif 'st' in self.args['model_name']:
                    self.model.embeddings.word_embeddings.weight[new_token_id].requires_grad = True
        self.vocab_size = len(self.tokenizer)
        self.model = self.model.to(self.device)
        self.prompt_model = Prompt_model(args, self.model, self.tokenizer, self.device, self.new_tokens)
        self.prompt_model.vocab_size = self.vocab_size
        self.max_norm = 0.1 if args['model_name'] in ['llama-7b','llama-7b-old'] else 10
        
        if args['model_name'] in ['llama-7b','llama-7b-old'] and args['mode'] in ['train','only_train','rank','rank_train','d']:
            if os.path.exists('/home/bingxing2'):
                peft_model_name = '/home/bingxing2/home/scx7140/.cache/huggingface/hub/models--castorini--repllama-v1-7b-lora-passage/snapshots/53f3ec543eafdc511092ade9de3f8b5a00155a3e'
            else:
                peft_model_name = '/data/home/scv6830/.cache/huggingface/hub/models--castorini--repllama-v1-7b-lora-passage/snapshots/53f3ec543eafdc511092ade9de3f8b5a00155a3e'
            base_model = self.model
            model = PeftModel.from_pretrained(base_model, peft_model_name)
            self.ranking_model = model.merge_and_unload()
            self.prompt_model.ranking_model = self.ranking_model
            self.get_query_embedding = self.prompt_model.get_query_embedding
        else:
            self.ranking_model = self.model
            self.prompt_model.ranking_model = self.model
            self.get_query_embedding = self.prompt_model.get_query_embedding_st
        
        if args['load_check_point']:
            self.load_check_point()
    
    def freeze_model(self,):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model_dict(self,):
        re = {'new_tokens':[]}
        for new_token in self.new_tokens:
            new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
            re['new_tokens'] = self.prompt_model.token_weights
        if self.args['enable_grad']:
            re['total_model'] = self.model.state_dict()
        
        if type(self.prompt_model.encoding_model) != list:
            re['encoding_model'] = self.prompt_model.encoding_model.state_dict()
        else:
            re['encoding_model'] = [item.state_dict() for item in self.prompt_model.encoding_model]
        return re
    
    # todo: it is difficult to calculate the uncertainty because the sequence has too many samples?
    # use the not rational setting and see if it works
    def get_entrophy(self,output, content_all_mask, content_all, content_true_mask, split=False):
        def entropy(p):
            # 过滤掉概率为0的事件，因为log(0)是未定义的
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        logits = output.logits[:, :-1, :] # b * seq_all-1 * logits
        content_all_mask = content_all_mask[:,1:]
        
        labels_mask = torch.zeros(content_all_mask.shape)
        content_true_mask_sum = torch.sum(content_true_mask, dim=1).int()
        content_all_mask_sum = torch.sum(content_all_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][content_all_mask_sum[batch_id]-content_true_mask_sum[batch_id]:content_all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to(self.device) # b * seq_true
        labels = content_all[:, :]
        if split:
            loss = []
            for batch_id in range(labels_mask.shape[0]):
                labels_tmp = labels[batch_id][content_true_mask[batch_id]==1]
                logits_tmp = logits[batch_id][labels_mask[batch_id]==1]
                loss.append(torch.nn.functional.cross_entropy(logits_tmp, labels_tmp, reduction='mean'))
        else:
            labels = labels[content_true_mask==1]
            logits = logits[labels_mask==1]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        return loss
    
    def entrophy(self, test_dataset, output_file, file_name=None):
        batch_size = 4 if self.args['model_name'] in ['llama-7b'] and self.args['batch_size'] > 4 else self.args['batch_size']
        test_dataloader = DataLoader(test_dataset,  batch_size, shuffle=False, num_workers=1)
        output_file = json.load(open(self.args['checkpoint_path']+'/'+output_file+'.json'))
        content_pred_new_list = np.array(output_file['content_pred'])
        content_true_new_list = np.array(output_file['content_true'])
        
        
        re = {'entrophy':[], 'data_id':[],'valid_loss':[]}
        self.prompt_model.eval()
        if self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']
        
        batch_id = 0
        for input_samples in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true,content_prev_mask,content_true_mask, content_all, content_all_mask, data_id = input_samples['content_prev'], input_samples['additional_bs'], input_samples['content_prev_sep'], input_samples['content_true'], input_samples['content_prev_mask'], input_samples['content_true_mask'], input_samples['content_all'], input_samples['content_all_mask'], input_samples['task_id']
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            data_id = data_id.numpy().tolist()
            
            content_pred_new = content_pred_new_list[batch_id:batch_id+batch_size]
            content_pred_new = torch.stack([self.tokenizer.encode_plus(item, max_length=32, truncation=True, return_tensors='pt', add_special_tokens = False, padding='max_length',)['input_ids'][0].to(self.device) for item in content_pred_new], dim=0)
            batch_id += batch_size
            
            # print(content_true)
            # print(content_pred_new)
            mask_zero_indices = (content_prev_mask == 0).long().argmax(dim=1)
            for i in range(content_prev.size(0)):
                zero_idx = mask_zero_indices[i]
                content_all[i][zero_idx:zero_idx+len(content_pred_new[i])] = content_pred_new[i]
            # # jiayudebug snippet
            # inputs = ''
            # while inputs != 'continue':
            #     try:
            #         print(eval(inputs))
            #     except Exception as e:
            #         print('error:', e)
            #         pass
            #     inputs = input()
            # mask_zero_indices = (content_all_mask == 0).long().argmax(dim=1)
            # for i in range(content_prev.size(0)):
            #     zero_idx = mask_zero_indices[i]
            #     content_all_mask[i][zero_idx:zero_idx+5] = 1
            # # jiayudebug snippet
            # inputs = ''
            # while inputs != 'continue':
            #     try:
            #         print(eval(inputs))
            #     except Exception as e:
            #         print('error:', e)
            #         pass
            #     inputs = input()
            # we should replace content_all with the new information
            
            if self.args['input_method'] == 'without_text':
                output, content_all_mask = self.prompt_model(content_true, content_true_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
            else:
                output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
            content_pred_new = content_pred_new[:,:content_true.shape[1]]
            loss_list = self.get_loss(output, content_all_mask, content_pred_new, content_true_mask, split=True, total_split=True) 
            for loss in loss_list:
                re['entrophy'].append(loss.detach().cpu().numpy().tolist())
                
        for input_samples in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true,content_prev_mask,content_true_mask, content_all, content_all_mask, data_id = input_samples['content_prev'], input_samples['additional_bs'], input_samples['content_prev_sep'], input_samples['content_true'], input_samples['content_prev_mask'], input_samples['content_true_mask'], input_samples['content_all'], input_samples['content_all_mask'], input_samples['task_id']
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            data_id = data_id.numpy().tolist()
            if self.args['input_method'] == 'without_text':
                output, content_all_mask = self.prompt_model(content_true, content_true_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
            else:
                output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)  
            loss_list = self.get_loss(output, content_all_mask, content_true, content_true_mask, split=True, total_split=True) 
            for loss in loss_list:
                re['valid_loss'].append(loss.detach().cpu().numpy().tolist())
                
        
        if file_name is not None:
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+'.json', 'w'))

    def get_loss(self, output, content_all_mask, content_all, content_true_mask, split=False,total_split=False):
        logits = output.logits[:, :-1, :] # b * seq_all-1 * logits
        content_all_mask = content_all_mask[:,1:]
        
        labels_mask = torch.zeros(content_all_mask.shape)
        content_true_mask_sum = torch.sum(content_true_mask, dim=1).int()
        content_all_mask_sum = torch.sum(content_all_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][content_all_mask_sum[batch_id]-content_true_mask_sum[batch_id]:content_all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to(self.device) # b * seq_true
        labels = content_all[:, :]
        if split:
            loss = []
            for batch_id in range(labels_mask.shape[0]):
                labels_tmp = labels[batch_id][:len(content_true_mask[batch_id])][content_true_mask[batch_id]==1]
                logits_tmp = logits[batch_id][labels_mask[batch_id]==1]
                loss.append(torch.nn.functional.cross_entropy(logits_tmp, labels_tmp, reduction='mean' if total_split == False else 'none'))
        else:
            labels = labels[content_true_mask==1]
            logits = logits[labels_mask==1]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        # loss /= content_true.shape[1]
        return loss

    def load_check_point(self, path=None):
        if path is None:
            path = f'{self.args["checkpoint_path"]}/model.pt'
        re = torch.load(path, map_location=torch.device('cpu'))
        if self.args['enable_grad']:
            self.model.load_state_dict(re['total_model'])
        else:
            self.prompt_model.token_weights.data = re['new_tokens'].detach()
                
        self.check_point = re

    def get_distribute_loss(self, output, content_all_mask, content_all, content_true_mask, split=False, top_k = 100):
        logits = output.logits[:, :-1, :] # b * seq_all-1 * logits
        content_all_mask = content_all_mask[:,1:]
        
        labels_mask = torch.zeros(content_all_mask.shape)
        content_true_mask_sum = torch.sum(content_true_mask, dim=1).int()
        content_all_mask_sum = torch.sum(content_all_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][content_all_mask_sum[batch_id]-content_true_mask_sum[batch_id]:content_all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to(self.device) # b * seq_true
        labels = content_all[:, :]
        info = []
        if split:
            loss = []
            for batch_id in range(labels_mask.shape[0]):
                labels_tmp = labels[batch_id][content_true_mask[batch_id]==1]
                logits_tmp = logits[batch_id][labels_mask[batch_id]==1]
                values, indices = torch.topk(logits_tmp, dim=1, k = top_k)
                new_info = [indices.detach().cpu().numpy().tolist()]
                new_info.append((torch.argsort(logits_tmp,dim=1, descending=True) == labels_tmp.unsqueeze(1)).nonzero(as_tuple=True)[1].detach().cpu().numpy().tolist())
                info.append(new_info)
                loss.append(torch.nn.functional.cross_entropy(logits_tmp, labels_tmp, reduction='mean'))
        else:
            labels = labels[content_true_mask==1]
            logits = logits[labels_mask==1]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        # loss /= content_true.shape[1]
        return loss, info

    def test_distribution(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(test_dataset, batch_size = 4 if self.args['model_name'] in ['llama-7b'] and self.args['batch_size'] > 4 else self.args['batch_size'] , shuffle = False, num_workers =1)
        re = {'valid_loss':[], 'content_pred':[], 'content_true':[], 'content_prev':[],'content_pred_token_ids':[],'content_prev_tokens_length':[], 'info':[]}
        self.prompt_model.eval()
        if self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            
            output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False,mode='test')
            loss_list, info = self.get_distribute_loss(output, content_all_mask, content_true, content_true_mask, split=True) 
            for loss in loss_list:
                re['valid_loss'].append(loss.item())
            for item in info:
                re['info'].append(info)
            if len(re['valid_loss']) > 10 and self.args['mode'] in ['train','evaluate_test']:
                break

        if file_name is not None:
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+'.json', 'w'))

    def test(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(test_dataset, batch_size = 4 if self.args['model_name'] in ['llama-7b'] and self.args['batch_size'] > 4 else self.args['batch_size'] , shuffle=False, num_workers=1)
        re = {'valid_loss':[], 'content_pred':[], 'content_true':[], 'content_prev':[],'content_pred_token_ids':[],'content_prev_tokens_length':[],'data_id':[]}
        self.prompt_model.eval()
        if self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']
        for input_samples in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true,content_prev_mask,content_true_mask, content_all, content_all_mask, data_id = input_samples['content_prev'], input_samples['additional_bs'], input_samples['content_prev_sep'], input_samples['content_true'], input_samples['content_prev_mask'], input_samples['content_true_mask'], input_samples['content_all'], input_samples['content_all_mask'], input_samples['task_id']
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            all_predicted_tokens = self.prompt_model.generate(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep,mode='test')
            data_id = data_id.numpy().tolist()
            for i in range(content_all.shape[0]):
                re['content_true'].append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_true[i])).replace('<|endoftext|>','').replace('⁇','').replace('</s>','').replace('<unk>','').strip())
                predicted_tokens = all_predicted_tokens[i]
                try:
                    content_pred_tokens = self.tokenizer.convert_ids_to_tokens(predicted_tokens)
                except:
                    content_pred_tokens = []
                    for item in predicted_tokens:
                        try:
                            content_pred_tokens.append(self.tokenizer.convert_ids_to_tokens([item])[0])
                        except:
                            continue    
                re['content_pred_token_ids'].append([item.detach().cpu().numpy().tolist() for item in predicted_tokens])
                re['content_pred'].append(self.tokenizer.convert_tokens_to_string(content_pred_tokens))
                re['content_prev'].append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_prev[i])).replace('<|endoftext|>','').replace('⁇','').replace('</s>','').replace('<unk>','').strip())
                re['data_id'].append(data_id[i])
                # content length有bug!
                re['content_prev_tokens_length'].append(float(torch.sum(content_prev_mask[i]).detach().cpu().numpy()))
            if self.args['input_method'] == 'without_text':
                output, content_all_mask = self.prompt_model(content_true, content_true_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
            else:
                output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
            loss_list = self.get_loss(output, content_all_mask, content_true, content_true_mask, split=True) 
            for loss in loss_list:
                re['valid_loss'].append(loss.item())
            if len(re['valid_loss']) > 10 and self.args['mode'] in ['train','evaluate_test']:
                break

        if file_name is not None:
            with open(self.args['checkpoint_path']+'/'+file_name+'.txt', 'w') as f:
                for i in range(len(re['valid_loss'])):
                    f.write(re['content_prev'][i]+'\n')
                    f.write('content_pred: '+re['content_pred'][i] + '\n')
                    f.write('content_true: '+re['content_true'][i] + '\n')
                    f.write('-----------------------------\n')
            
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+'.json', 'w'))
    
    def save_document_tokens(self, dataset):
        test_dataloader = DataLoader(dataset, self.args['batch_size'] , shuffle=False, num_workers=1)
        fpid2embedding = {}
        for input_samples in tqdm.tqdm(test_dataloader, mininterval=300):
            for i in range(len(input_samples['fpid'])):
                fpid2embedding[input_samples['fpid'][i].detach().cpu().numpy().tolist()] = input_samples['token_possibility'][i].detach().cpu().numpy().tolist()
        return fpid2embedding
    
    def pre_train(self, dataset, dataloader, optimizer, parameters):
        
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, parameters), lr=self.args['pretrain_lr'])
        total_additional_loss = 0
        for input_samples in tqdm.tqdm(dataloader, mininterval=300):
            # content_prev, additional_bs, content_prev_sep, content_true,content_prev_mask,content_true_mask, content_all, content_all_mask, data_id, passage_embeddings
            content_prev, additional_bs,content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(input_samples['content_prev'],input_samples['additional_bs'], input_samples['content_prev_sep'], input_samples['content_true'], input_samples['content_prev_mask'], input_samples['content_true_mask'])   
            additional_loss = self.prompt_model.additional_loss(content_prev, content_prev_mask, additional_bs)
            total_additional_loss += additional_loss.item()
            optimizer.zero_grad()
            additional_loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, parameters), max_norm=self.max_norm)
            optimizer.step()
            
        return total_additional_loss / len(dataset)

    @torch.no_grad()
    def _compute_negative_mask(self, qids, docids):
        negative_mask = torch.zeros((len(qids), len(docids)), dtype=torch.bool, device=qids.device)
        for i, qid in enumerate(qids):
            negative_mask[i] = torch.logical_or(negative_mask[i], docids==qid)
        negative_mask = negative_mask.type(torch.float32)
        negative_mask.fill_diagonal_(0)
        return negative_mask

    # def get_ranking_loss(self, query_embeds, doc_ids, doc_embeds, neg_docids, neg_doc_embeds):
    #     loss = 0
    #     for i in range(len(query_embeds)):
    #         labels = torch.tensor(0, dtype=torch.long, device=query_embeds.device)
    #         all_doc_embeds = torch.vstack((doc_embeds[i].reshape(1,-1), neg_doc_embeds[i].reshape(3,-1),))
    #         similarities = torch.matmul(query_embeds[i], all_doc_embeds.transpose(0, 1))
    #         loss += nn.functional.cross_entropy(similarities, labels)
    #     return loss

    def get_ranking_loss(self, query_embeds, doc_ids, doc_embeds, neg_docids, neg_doc_embeds):
        neg_docids = torch.squeeze(neg_docids).reshape(-1,)
        labels = torch.arange(len(query_embeds), dtype=torch.long, device=query_embeds.device)
        all_doc_embeds = torch.vstack((doc_embeds, neg_doc_embeds.reshape(-1,neg_doc_embeds.shape[-1])),)
        all_docids = torch.hstack((doc_ids, neg_docids))
        negative_mask = self._compute_negative_mask(doc_ids, all_docids)
        similarities = torch.matmul(query_embeds, all_doc_embeds.transpose(0, 1)) # (batch_size * embedding_size) * (embedding_size * batch_size) = batch_size * batch_size
        similarities = similarities * self.inv_temperature
        similarities = similarities - 10000.0 * negative_mask
        loss = nn.functional.cross_entropy(similarities, labels)
        return loss
    
    def get_token_loss(self, output, token_possibility,):
        criterion = torch.nn.KLDivLoss()
        loss = criterion(output, token_possibility,) * 1e3
        return loss

    # torch.argmax(output.logits[i][mask_zero_indices[i]-1])

    def train(self, train_dataset, valid_dataset, test_dataset=None):  
        test_dataloader = DataLoader(test_dataset, batch_size = self.args['batch_size'], shuffle=False, num_workers=1) if test_dataset is not None else None
        train_dataloader = DataLoader(train_dataset, batch_size = self.args['batch_size'], shuffle=True, num_workers=1) 
        # train_dataloader = DataLoader(train_dataset, batch_size = self.args['batch_size'], shuffle=False, num_workers=1) 
        valid_dataloader = DataLoader(valid_dataset, batch_size = self.args['batch_size'], shuffle=True, num_workers=1) 
        
        best_loss = 100000000000
        # 改了early stop
        early_stop = 10
        min_epoch = 10
        early_stop_epochs = 0
        parameters = []
        parameters += self.prompt_model.parameters()
        if type(self.prompt_model.encoding_model) == list:
            for k in range(len(self.prompt_model.encoding_model)):
                parameters += self.prompt_model.encoding_model[k].parameters()
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=self.args['lr'], weight_decay=self.args['l2']) if self.args['model_name'] not in ['llama-7b','llama-7b-old'] else Adam16(filter(lambda p: p.requires_grad, parameters), lr=self.args['lr'], weight_decay=self.args['l2'])

        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args['weight_decay'])

        pretrain_optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=self.args['pretrain_lr'], weight_decay=self.args['l2']) if self.args['model_name'] not in ['llama-7b','llama-7b-old'] else Adam16(filter(lambda p: p.requires_grad, parameters), lr=self.args['pretrain_lr'], weight_decay=self.args['l2'])
        
        for epoch in range(self.args['pretrain_epochs']):
            self.prompt_model.train()
            total_loss = self.pre_train(train_dataset, train_dataloader, pretrain_optimizer, parameters)
            valid_loss = self.pre_train(valid_dataset, valid_dataloader, pretrain_optimizer, parameters)
            
            if test_dataloader is not None:
                self.pre_train(test_dataset, test_dataloader, optimizer, pretrain_optimizer, parameters)
            output_str = f"Pretraining Epoch {epoch}: Trainning Loss = {total_loss:.3f} Validation Loss = {valid_loss:.3f}"
            with open(self.args['checkpoint_path']+'/'+'log'+'.txt', 'a') as fw:
                fw.write(output_str+'\n')
            if self.args['wandb'] != 'none':
                wandb.log({"pre_train Trainning Loss": total_loss, "pre_train Validation Loss": valid_loss})
            
        for epoch in range(self.args['num_epochs']):
            total_loss = 0
            self.prompt_model.train()
            if 'Narratives' in self.args['task_name'] and 'person' not in self.args['task_name']:
                random.shuffle(train_dataset.inputs)
                train_dataloader = DataLoader(train_dataset, batch_size = self.args['batch_size'], shuffle=True, num_workers=1) 
            for input_samples in tqdm.tqdm(train_dataloader, mininterval=300):
                torch.cuda.empty_cache()
                content_prev, additional_bs, content_prev_sep, content_true,content_prev_mask,content_true_mask, content_all, content_all_mask = input_samples['content_prev'], input_samples['additional_bs'], input_samples['content_prev_sep'], input_samples['content_true'], input_samples['content_prev_mask'], input_samples['content_true_mask'], input_samples['content_all'], input_samples['content_all_mask']
                content_prev, additional_bs,content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev,additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)   
                content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
                if self.args['mode'] == 'generation_train':
                    if self.args['input_method'] == 'without_text':
                        output, content_all_mask2 = self.prompt_model(content_true, content_true_mask, additional_bs, additional_bs_mask, content_prev_sep,)
                    else:
                        output, content_all_mask2 = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep,)
                    loss = self.get_loss(output, content_all_mask2, content_true, content_true_mask)
                elif self.args['mode'] == 'token_train':
                    output = self.prompt_model.forward_token(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep)
                    token_possibility = input_samples['token_possibility'].clone()
                    token_possibility = token_possibility.to(self.device)
                    loss = self.get_token_loss(output, token_possibility,)
                    del token_possibility
                    gc.collect()
                    torch.cuda.empty_cache()
                # get ranking loss
                else:
                    query_embeddings = self.get_query_embedding(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep,mode='train')
                    # loss = self.get_ranking_loss(query_embeddings, pos_pid.to(self.device),passage_embeddings.to(self.device), neg_pid.to(self.device), neg_p_embedding.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, parameters), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item()
            
            total_loss /= len(train_dataset)

            if self.args['evaluate_log']:
                self.test(valid_dataset, f'test_{epoch}')

            valid_loss = 0
            self.prompt_model.eval()
            for input_samples in tqdm.tqdm(valid_dataloader, mininterval=300):
                torch.cuda.empty_cache()
                content_prev, additional_bs, content_prev_sep, content_true,content_prev_mask,content_true_mask, content_all, content_all_mask = input_samples['content_prev'], input_samples['additional_bs'], input_samples['content_prev_sep'], input_samples['content_true'], input_samples['content_prev_mask'], input_samples['content_true_mask'], input_samples['content_all'], input_samples['content_all_mask']
                content_prev, additional_bs,content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev,additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
                content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
                if self.args['mode'] == 'generation_train':
                    if self.args['input_method'] == 'without_text':
                        output, content_all_mask = self.prompt_model(content_true, content_true_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
                    else:
                        output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False)
                    generation_loss = self.get_loss(output, content_all_mask, content_true, content_true_mask)
                    valid_loss += generation_loss.item()
                elif self.args['mode'] == 'token_train':
                    output = self.prompt_model.forward_token(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep)
                    token_possibility = input_samples['token_possibility'].clone()
                    token_possibility = token_possibility.to(self.device)
                    # # token_possibility = input_samples['token_possibility'].clone().to(self.device)
                    valid_loss += self.get_token_loss(output, token_possibility,).item()
                    del token_possibility
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    query_embeddings = self.get_query_embedding(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep,mode='train')
                    # ranking_loss = self.get_ranking_loss(query_embeddings, pos_pid.to(self.device),passage_embeddings.to(self.device), neg_pid.to(self.device), neg_p_embedding.to(self.device))
                    # valid_loss += ranking_loss.item()
            valid_loss /= len(valid_dataset)
            output_str = f"Epoch {epoch}: Trainning Loss = {total_loss:.3f} Validation Loss = {valid_loss:.3f}"
            if self.args['wandb'] != 'none':
                wandb.log({"Trainning Loss": total_loss, "Validation Loss": valid_loss})
            with open(self.args['checkpoint_path']+'/'+'log'+'.txt', 'a') as fw:
                fw.write(output_str+'\n')
            print(output_str)
            if valid_loss < best_loss:
                best_loss = valid_loss
                early_stop_epochs = 0
                best_model_wts = self.get_model_dict()
                print(f'get better model in epoch {epoch}, saved')
                torch.save(best_model_wts, self.args['checkpoint_path']+'/model.pt')
            else:
                early_stop_epochs += 1
                if early_stop_epochs >= early_stop and epoch > min_epoch:
                    print(f'early stop at epoch {epoch}')
                    with open(self.args['checkpoint_path']+'/'+'log'+'.txt', 'a') as fw:
                        fw.write(f'early stop at epoch {epoch}'+'\n')
                    break
            if self.args['weight_decay'] < 1 and epoch <= 10:
                scheduler.step()


if __name__ == '__main__':
    args = {'model_name':'gpt2','brain_embed_size':1000, 'word_embed_size':768,'cuda':5}
    decoding_model = Decoding_model(args)


