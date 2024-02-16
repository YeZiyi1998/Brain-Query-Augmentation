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
import tqdm

def get_embeddings_llama(query, model, tokenizer, input_type='query', device = None):
    if input_type == 'query':
        query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
    elif input_type == 'passage':
        query_input = tokenizer(f'passage: {query}</s>', return_tensors='pt')
    query_input.to(device)
    with torch.no_grad():
        # compute query embedding
        query_outputs = model(**query_input,)
        query_embedding = query_outputs.last_hidden_state[0][-1]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).detach().cpu().numpy()
    return query_embedding

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings_st(query, model, tokenizer, input_type='query', device = None):
    query_input = tokenizer(query, return_tensors='pt')
    query_input.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**query_input, return_dict=True)
    # Perform pooling
    embeddings = mean_pooling(model_output, query_input['attention_mask']).detach().cpu().numpy()
    return embeddings

def get_embeddings_None(query, model, tokenizer, input_type='query', device = None):
    return -1

def get_embeddings_bingxing(queries, model, tokenizer, input_type='query', batch_size=8, device=None):
    all_embeddings = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        if input_type == 'query':
            queries_input = tokenizer([f'query: {q}</s>' for q in batch_queries], padding=True, truncation=True, return_tensors='pt')
        elif input_type == 'passage':
            queries_input = tokenizer([f'passage: {q}</s>' for q in batch_queries], padding=True, truncation=True, return_tensors='pt')
        queries_input.to(device)
        lengths = queries_input['attention_mask'].sum(dim=1) - 1 
        with torch.no_grad():
            # 计算嵌入向量
            outputs = model(**queries_input)
            embeddings = outputs.last_hidden_state[torch.arange(len(lengths)), lengths]
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(normalized_embeddings)
    # 合并所有批次的结果
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

class FMRI_dataset_old():
    def pack_info(self, content_prev, additional_bs, content_true, trail_id,task_id, passage_embedding=None, passage_content=None,query_embedding=None,did=None,pid=None,k=None,fpid=None,document_embedding=None,document_content=None,fdid=None):
        query = content_prev
        perceived = content_true
        if self.args['model_name'] in ['llama-7b',]:
            self.add_special_tokens = True
            content_all = self.tokenizer.encode_plus(content_prev+' '+content_true, max_length=64, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length',)
            content_all['input_ids'] = content_all['input_ids'][:,1:]
            content_all['attention_mask'] = content_all['attention_mask'][:,1:]
            content_true = self.tokenizer.encode_plus(content_true.strip(), max_length=32, truncation=True, return_tensors='pt',padding='max_length',)
            content_true['input_ids'] = content_true['input_ids'][:,1:]
            content_true['attention_mask'] = content_true['attention_mask'][:,1:]
            content_prev = self.tokenizer.encode_plus(content_prev, max_length=32, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length', )
            content_prev['input_ids'] = content_prev['input_ids'][:,1:]
            content_prev['attention_mask'] = content_prev['attention_mask'][:,1:]
        else:
            content_all = self.tokenizer.encode_plus(content_prev+' '+content_true, max_length=64, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens, padding='max_length')
            content_true = self.tokenizer.encode_plus(' '+content_true, max_length=32, truncation=True, return_tensors='pt',padding='max_length')
            content_prev = self.tokenizer.encode_plus(content_prev, max_length=32, truncation=True, return_tensors='pt', add_special_tokens = self.add_special_tokens or self.args['model_name'] == 'st', padding='max_length')
        return {
                'content_prev': content_prev['input_ids'][0], 
                'content_prev_mask': content_prev['attention_mask'][0], 
                'additional_bs':torch.tensor(additional_bs, dtype=torch.float32),
                'content_prev_sep':self.tokenizer.encode_plus(['<brain/>', '</brain>'],return_tensors='pt')['input_ids'][0], 
                'content_true': content_true['input_ids'][0], 
                'content_true_mask': content_true['attention_mask'][0], 
                'trail_id': trail_id,
                'content_all': content_all['input_ids'][0],
                'content_all_mask': content_all['attention_mask'][0],
                'task_id':task_id,
                'passage_embedding': passage_embedding,
                'query_content': query,
                'perceived': perceived,
                'passage_content': passage_content,
                'query_embedding': query_embedding,
                'pid':pid,
                'did':did,
                'k':k,
                'fpid':fpid,
                'document_embedding': document_embedding,
                'document_content': document_content,
                'fdid': fdid
            }
    
    def normalized(self, dic_pere):
        all_data = []
        for story in dic_pere.keys():
            all_data.append(np.array(dic_pere[story]['fmri']))
        all_data = np.concatenate(all_data, axis=0) 
        all_data = self.scaler.fit_transform(all_data)
        start_idx = 0
        for story in dic_pere.keys():
            dic_pere[story]['fmri'] = all_data[start_idx:start_idx+dic_pere[story]['fmri'].shape[0]]
            start_idx += dic_pere[story]['fmri'].shape[0] 
    
    def split_Pereira(self, input_dataset):
        cid2content = {}
        for sid, story in enumerate(input_dataset.keys()):
            cid2content[sid] = {}
            for item_id, item in enumerate(input_dataset[story]):
                cid2content[sid][item_id] = {}
                for k in range(0, len(item['word'])):
                    cid2content[sid][item_id][k] = item['word'][k]['content'].strip()
        return cid2content
    
    def split_Huth(self, input_dataset, data_info2=None, split_num=10):
        info2cid = {}
        cid2content = {}
        content2cid = {}
        cid = 0
        for sid, story in enumerate(input_dataset.keys()):
            info2cid[sid] = {}
            if data_info2 is not None and story not in data_info2:
                continue
            for item_id, item in enumerate(input_dataset[story]):
                info2cid[sid][item_id] = {}
                for k in range(0, len(item['word'])):
                    content = item['word'][k]['content'].strip()
                    if content not in content2cid.keys():
                        content2cid[content] = cid
                        cid2content[cid] = content
                        info2cid[sid][item_id][k] = cid
                        cid += 1
                    else:
                        info2cid[sid][item_id][k] = content2cid[content]
        pid2cid = {}
        cid2pid = dict((k,k//split_num) for k in range(cid))
        for k, v in cid2pid.items():
            if v not in pid2cid.keys():
                pid2cid[v] = [k]
            else:
                pid2cid[v].append(k)
        return cid2content, pid2cid, cid2pid, info2cid
    
    def split_Narratives(self, args, subject_name, input_dataset,split_num=10):
        info2cid = {}
        cid2content = {}
        content2cid = {}
        cid = 0
        for sid, story in enumerate(args['Narratives_stories']):
            info2cid[sid] = {}
            subject = f'sub-{subject_name}'
            for item_id, item in enumerate(input_dataset[story][subject]):
                info2cid[sid][item_id] = {}
                for k in range(0, len(item['word'])):
                    content = item['word'][k]['content'].strip()
                    if content not in content2cid.keys():
                        content2cid[content] = cid
                        cid2content[cid] = content
                        info2cid[sid][item_id][k] = cid
                        cid += 1
                    else:
                        info2cid[sid][item_id][k] = content2cid[content]
                        
        pid2cid = {}
        cid2pid = dict((k,k//split_num) for k in range(cid))
        for k, v in cid2pid.items():
            if v not in pid2cid.keys():
                pid2cid[v] = [k]
            else:
                pid2cid[v].append(k)
        return cid2content, pid2cid, cid2pid, info2cid
    
    def __init__(self, input_dataset, args, tokenizer,decoding_model=None,prefix=None):
        self.decoding_model = decoding_model
        self.args = args
        self.add_special_tokens=False
        self.inputs = []
        self.shuffle_times = args['shuffle_times']
        dataset_path = args['dataset_path']
        if args['normalized']:
            self.scaler = StandardScaler()
        self.tokenizer = tokenizer
        task_id = 0
        self.fpm = {}
        self.fdm = {}
        model_name2get_embeddings_function = {
            'st': get_embeddings_st,
            'llama-7b': get_embeddings_llama,
            'gpt2-xl':get_embeddings_None,
            'gpt2':get_embeddings_None,
            'gpt2-large':get_embeddings_None,
        }
        get_embeddings =  model_name2get_embeddings_function[args['model_name']] 
        if 'Pereira' in args['task_name']:
            dataset_name, subject_name = args['task_name'].split('_')
            pere_dataset = pickle.load(open(f'{dataset_path}{subject_name}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'{dataset_path}{subject_name}.wq.pkl.dic','rb'))
            if args['normalized']:
                self.normalized(pere_dataset)
            cid2content = self.split_Pereira(input_dataset)
            fpid = 0
            fdid = 0
            content_true2idx = {}
            for sid, story in tqdm.tqdm(enumerate(input_dataset.keys())):
                fdid += 1
                full_document_content = ' '.join([cid2content[sid][tmp_item_id][j] for tmp_item_id in range(len(input_dataset[story])) for j in range(0,len(input_dataset[story][tmp_item_id]['word']))])
                full_document_embedding = get_embeddings(full_document_content,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device, input_type='passage')
                self.fdm[fdid] = {'full_document_content':full_document_content, 'full_document_embedding':full_document_embedding}
                # random split
                if args['data_spliting'] == 'random':
                    random_number = random.random()
                for item_id, item in enumerate(input_dataset[story]):
                    fpid += 1
                    full_passage_content = ' '.join([cid2content[sid][item_id][j] for j in range(0,len(item['word']))])
                    full_passage_embedding = get_embeddings(full_passage_content,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device, input_type='passage')
                    self.fpm[fpid] = {'full_passage_content':full_passage_content, 'full_passage_embedding':full_passage_embedding}
                    for k in range(1, len(item['word'])):
                        # generation information
                        content_prev = ' '.join([cid2content[sid][item_id][j] for j in range(0,k)][-15:])
                        additional_bs = np.array([pere_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = cid2content[sid][item_id][k]
                        if args['add_end']:
                            content_true += '<|endoftext|>'
                        if args['data_spliting'] == '0107':
                            if content_true not in content_true2idx.keys():
                                content_true2idx[content_true] = random.random()
                            random_number = content_true2idx[content_true]
                        # ranking information
                        tmp_cid = [j for j in range(0,k)]
                        tmp_passage_content = ' '.join([cid2content[sid][item_id][j] for j in range(0,len(item['word'])) if j not in tmp_cid])
                        tmp_document_content = ' '.join([cid2content[sid][tmp_item_id][j] for tmp_item_id in range(len(input_dataset[story])) for j in range(0,len(input_dataset[story][tmp_item_id]['word'])) if tmp_item_id != item_id or j not in tmp_cid])
                        document_embedding = get_embeddings(tmp_document_content,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device,input_type='passage')
                        query_embedding = get_embeddings(content_prev,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device)
                        passage_embedding = get_embeddings(tmp_passage_content,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device, input_type='passage')
                        packed_info = self.pack_info(content_prev, additional_bs, content_true, random_number, task_id = task_id, passage_embedding = passage_embedding, passage_content = tmp_passage_content,query_embedding=query_embedding,did=sid,pid=item_id,k=k,fpid=fpid,document_embedding=document_embedding,document_content=tmp_document_content,fdid=fdid)
                        task_id += 1
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)   
        elif 'Narratives' in args['task_name']:
            subject_name = args['task_name'].split('_')[1]
            content_true2idx = {}
            
            cid2content, pid2cid, cid2pid, info2cid = self.split_Narratives(args, subject_name, input_dataset, split_num = args['split_num'])
            pid2idx = {}
            for pid in pid2cid.keys():
                full_passage_content = ' '.join([cid2content[tmp_cid] for tmp_cid in pid2cid[pid]])
                full_passage_embedding = get_embeddings(full_passage_content.strip(),self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device,input_type='passage')
                self.fpm[pid] = {'full_passage_content':full_passage_content, 'full_passage_embedding':full_passage_embedding}
            
            for sid, story in enumerate(args['Narratives_stories']):
                Narratives_dataset = pickle.load(open(f'{dataset_path}{story}.pca1000.wq.pkl.dic','rb')) if args['fmri_pca'] else pickle.load(open(f'{dataset_path}{story}.wq.pkl.dic','rb'))
                for subject in [f'sub-{subject_name}']:
                    for item_id, item in enumerate(input_dataset[story][subject]):
                        for k in range(1, len(item['word'])):
                            cid = info2cid[sid][item_id][k]
                            content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                            content_prev = ' '.join(content_prev.split()[-15:])
                            query_cid_list = [info2cid[sid][item_id][j] for j in range(0,k)]
                            additional_bs = np.array([Narratives_dataset[subject]['fmri'][idx] for idx in item['word'][k]['additional']])
                            content_true = item['word'][k]['content']
                            p_id = cid2pid[cid]
                            if len(content_true.strip()) == 0:
                                continue
                            if args['add_end']:
                                content_true += '<|endoftext|>'
                            if args['data_spliting'] == '0107':
                                if content_true not in content_true2idx.keys():
                                    content_true2idx[content_true] = random.random()
                                trail_id = content_true2idx[content_true]
                            else:
                                if p_id not in pid2idx.keys():
                                    pid2idx[p_id] = random.random()
                                trail_id = pid2idx[p_id]
                            query_embedding = get_embeddings(content_prev,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device)
                            tmp_passage_content = ' '.join([cid2content[tmp_cid] for tmp_cid in pid2cid[p_id] if tmp_cid not in query_cid_list])
                            tmp_passage_embedding = get_embeddings(tmp_passage_content.strip(),self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device,input_type='passage',)
                            packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id, task_id = task_id, passage_embedding = tmp_passage_embedding, passage_content=tmp_passage_content,query_embedding=query_embedding,fpid=p_id,did=sid,pid=item_id,k=k,)
                            task_id += 1
                            if len(packed_info['content_true']) > 0 and len(packed_info['content_prev']) > 0:
                                self.inputs.append(packed_info)
        elif 'Huth' in args['task_name']:
            dataset_name = args['task_name'].split('_')[0]
            subject_name = args['task_name'].split('_')[1]
            data_info2 = json.load(open('../dataset_info/Huth.json'))
            cid2content, pid2cid, cid2pid, info2cid = self.split_Huth(input_dataset, data_info2, split_num = args['split_num'])
            ds_dataset = pickle.load(open(f'{dataset_path}{subject_name}.pca1000.wq.pkl.dic','rb'))
            pid2idx = {}
            for pid in pid2cid.keys():
                full_passage_content = ' '.join([cid2content[tmp_cid] for tmp_cid in pid2cid[pid]])
                full_passage_embedding = get_embeddings(full_passage_content.strip(),self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device,input_type='passage')
                self.fpm[pid] = {'full_passage_content':full_passage_content, 'full_passage_embedding':full_passage_embedding}
            content_true2idx = {}
            for sid, story in tqdm.tqdm(enumerate(input_dataset.keys()), mininterval=300):
                if story not in data_info2:
                    continue
                for item_id, item in enumerate(input_dataset[story]):
                    for k in range(1, len(item['word'])):
                        cid = info2cid[sid][item_id][k]
                        content_prev = ' '.join([item['word'][j]['content'] for j in range(0,k)])
                        query_cid_list = [info2cid[sid][item_id][j] for j in range(0,k)]
                        content_prev = ' '.join(content_prev.split()[-15:])
                        additional_bs = np.array([ds_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = item['word'][k]['content']
                        if len(content_true.strip()) == 0:
                            continue
                        p_id = cid2pid[cid]
                        if args['add_end']:
                            content_true += '<|endoftext|>'
                        if args['data_spliting'] == '0107':
                            if content_true not in content_true2idx.keys():
                                content_true2idx[content_true] = random.random()
                            trail_id = content_true2idx[content_true]
                        else:
                            if p_id not in pid2idx.keys():
                                pid2idx[p_id] = random.random()
                            trail_id = pid2idx[p_id]
                        query_embedding = get_embeddings(content_prev,self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device)
                        tmp_passage_content = ' '.join([cid2content[tmp_cid] for tmp_cid in pid2cid[p_id] if tmp_cid not in query_cid_list])
                        tmp_passage_embedding = get_embeddings(tmp_passage_content.strip(),self.decoding_model.ranking_model,self.tokenizer,device = self.decoding_model.device,input_type='passage',)
                        packed_info = self.pack_info(content_prev, additional_bs, content_true, trail_id, task_id = task_id, passage_embedding = tmp_passage_embedding, passage_content=tmp_passage_content,query_embedding=query_embedding,fpid=p_id,did=sid,pid=item_id,k=k,)
                        task_id += 1
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)
                        if self.inputs[-1]['content_prev'].shape[0] == 0 and args['context']:
                            self.inputs = self.inputs[:-1]
        pickle.dump(self.fpm, open(f'../../../common_data/{args["task_name"].split("/")[-1]}/{prefix}fpm.pkl','wb'))
        pickle.dump(self.fdm, open(f'../../../common_data/{args["task_name"].split("/")[-1]}/{prefix}fdm.pkl','wb'))
        