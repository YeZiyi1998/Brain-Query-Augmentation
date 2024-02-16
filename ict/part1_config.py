import argparse
import json
import os
import pickle
import numpy as np
np.random.seed(2021)

mode2info = {
    'passage':'fpid',
    'document':'fdid'
}

def random_selection(strings, num_selections):
    if len(strings) < num_selections:
        return strings
    selected_strings = np.random.choice(strings, num_selections, replace=False)
    return selected_strings

# load command args
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cut_off', default=5, type=int)
    parser.add_argument('--query_length', default=1e6, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--model', default='[repllama]', type=str)
    parser.add_argument('--task_name', default='Huth_1', type=str)
    parser.add_argument('--result_path', default='None', type=str)
    parser.add_argument('--record', default='False', type=str)
    parser.add_argument('--split_num', default=10, type=int)
    parser.add_argument('--test_ids', default='0.2,0.4', type=str)
    parser.add_argument('--common_data_prefix', default='', type=str)
    parser.add_argument('--doc_collection', default='all', choices = ['all', ''], type=str)
    parser.add_argument('--result_file_name', default='', type=str)
    parser.add_argument('--query_sample', default='False', type=str)
    parser.add_argument('--smaller_docs', default='False', type=str)
    parser.add_argument('--entrophy', default='False', type=str)
    parser.add_argument('--sample', default='False', type=str)
    parser.add_argument('--all_random_number', default='False', type=str)
    parser.add_argument('--valid_loss', default='False', type=str)
    args = parser.parse_args()
    args.valid_loss = args.valid_loss == "True"
    args.record = args.record == "True"
    args.entrophy = args.entrophy == 'True'
    args.all_random_number = args.all_random_number == "True"
    args.query_sample = args.query_sample == "True"
    args.sample = args.sample == "True"
    args.smaller_docs = args.smaller_docs == "True"
    args.test_ids = [float(item) for item in args.test_ids.split(',')]
    args.model = [item for item in args.model.split(',')]
    base_path = ''
    # load the language generation's experimental results
    if args.result_path == "None":
        args.result_path = f'{base_path}/language_generation/results/{args.task_name}'
    else:
        args.result_path = f'{base_path}{args.result_path}'
    
    return args
    
def load_data(args):
    if args.all_random_number:
        return load_all_data(args)
    # load stimuli information
    all_input_data = pickle.load(open(f'../common_data/{args.task_name}/{args.common_data_prefix}all_input_data.pkl','rb'))
    if '10' not in args.common_data_prefix:
        print('loading data mask ...')
        all_input_data_mask = pickle.load(open(f'../common_data/{args.task_name}/llama-7b.10all_input_data.pkl','rb'))
    else:
        all_input_data_mask = all_input_data
    fpm = pickle.load(open(f'../common_data/{args.task_name}/{args.common_data_prefix}fpm.pkl','rb')) if 'Pereira' not in args.task_name else pickle.load(open(f'../common_data/{args.task_name}/{args.common_data_prefix}fdm.pkl','rb'))
    # building index for fpm
    all_input_data = [item for idx, item in enumerate(all_input_data) if all_input_data_mask[idx]['trail_id'] > args.test_ids[0] and all_input_data_mask[idx]['trail_id'] <= args.test_ids[1]] 
    
    # if 'train' not in args.model[0] else [item for item in all_input_data if item['trail_id'] <= args.test_ids[0] or item['trail_id'] > args.test_ids[1]]
    
    mode = 'document' if 'Pereira' in args.task_name else 'passage'
    return all_input_data, fpm, mode

def load_all_data(args):
    all_input_data_all = []
    all_input_data = pickle.load(open(f'../common_data/{args.task_name}/{args.common_data_prefix}all_input_data.pkl','rb'))
    if '10' not in args.common_data_prefix:
        print('loading data mask ...')
        all_input_data_mask = pickle.load(open(f'../common_data/{args.task_name}/llama-7b.10all_input_data.pkl','rb'))
    else:
        all_input_data_mask = all_input_data
        # building index for fpm
    for random_number in range(5):
        # load stimuli information
        margin = (random_number - 1) * 0.2
        tmp_all_input_data = [item for idx, item in enumerate(all_input_data) if all_input_data_mask[idx]['trail_id'] > args.test_ids[0] + margin and all_input_data_mask[idx]['trail_id'] <= args.test_ids[1] + margin] 
        all_input_data_all += tmp_all_input_data
    mode = 'document' if 'Pereira' in args.task_name else 'passage'
    fpm = pickle.load(open(f'../common_data/{args.task_name}/{args.common_data_prefix}fpm.pkl','rb')) if 'Pereira' not in args.task_name else pickle.load(open(f'../common_data/{args.task_name}/{args.common_data_prefix}fdm.pkl','rb'))
    return all_input_data_all, fpm, mode

def add2dic(dic1, dic2):
    for k in dic1.keys():
        dic1[k] += dic2[k]

def load_result_data(args, file_name):
    for random_number in range(5):
        if random_number != 1:
            tmp_result_path = args.result_path + f'_{random_number}'
        else:
            tmp_result_path = args.result_path 
        if random_number == 0:
            result = json.load(open(os.path.join(tmp_result_path, file_name)))
        else:
            add2dic(result, json.load(open(os.path.join(tmp_result_path, file_name))))
    return result
        
def load_entrophy(args, file_name):
    if args.all_random_number == False:
        return json.load(open(os.path.join(args.result_path, file_name)))
    for random_number in range(5):
        if random_number != 1:
            tmp_result_path = args.result_path + f'_{random_number}'
        else:
            tmp_result_path = args.result_path 
        if random_number == 0:
            result = json.load(open(os.path.join(tmp_result_path, file_name)))
        else:
            add2dic(result, json.load(open(os.path.join(tmp_result_path, file_name))))
    return result
    
