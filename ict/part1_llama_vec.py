# from model_utils.model_config import llama_path, get_embeddings, get_score, get_model, model_path, AutoModel, AutoTokenizer, my_token, get_reranking_score, get_ranking_model, rank_model_path, get_embeddings_bingxing
import argparse
import json
import os
import tqdm
from model_utils.evaluation import evaluate, print_performance, get_result_template
from part1_config import load_config, mode2info, load_data, random_selection, load_result_data, load_entrophy
from my_faiss import build_index, retrieve_from_index, merge_index
import numpy as np
import pickle
from bm25 import transform2dict, BM25, rm3_expansion
import copy
from multiprocessing.pool import Pool  
from my_woosh import get_woosh_bm25_result, rank_with_my_woosh, built_index
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring

def get_repllama_ranking_result(result, all_input_data, fpm, args, mode=None):
    ndcg_score_story_all = get_result_template()
    id_name, content_name = mode2info[mode], mode
    index = build_index(list([np.squeeze(item[f'full_{content_name}_embedding']).tolist() for item in fpm.values()]), list(fpm.keys()), dimension=list(fpm.values())[0][f'full_{content_name}_embedding'].shape[-1])

    if len(result['data_id']) != len(all_input_data):
        print('error! input data and output data with different length')

    for i in tqdm.tqdm(range(len(result['data_id'])), mininterval=300): # 1000
        query = all_input_data[i]['query_content']
        if len(query.split()) > args.query_length:
            continue
        target_p_id = all_input_data[i][id_name]
        enhanced_query = np.array(result['query_embedding'][i])
        y_pred_list = []
        true_list = []
        D, I = retrieve_from_index(index, enhanced_query, )
        newD, newI = [], []
        newD.append(np.dot(enhanced_query, np.squeeze(all_input_data[i][f'{content_name}_embedding'])))
        newI.append(target_p_id)
        y_pred_list, true_list = merge_index(D, I, newD, newI)
        evaluate(true_list, y_pred_list, ndcg_score_story_all)
    print_performance(ndcg_score_story_all)
    return ndcg_score_story_all

def get_repllama_result(result, all_input_data, fpm, args,mode=None):
    ndcg_score_story_all = get_result_template()

    id_name, content_name = mode2info[mode], mode
    index = build_index(list([np.squeeze(item[f'full_{content_name}_embedding']).tolist() for item in fpm.values()]), list(fpm.keys()), dimension=list(fpm.values())[0][f'full_{content_name}_embedding'].shape[-1])

    if len(result['query_embedding']) != len(all_input_data):
        print('error! input data and output data with different length')

    for i in tqdm.tqdm(range(len(result['query_embedding'])), mininterval=300): # 1000
        query = all_input_data[i]['query_content']
        if len(query.split()) > args.query_length:
            continue
        target_p_id = all_input_data[i][id_name]
        enhanced_query = np.array(result['query_embedding'][i])
        D, I = retrieve_from_index(index, enhanced_query, )
        newD, newI = [], []
        # newD.append(np.dot(enhanced_query, fpm[target_p_id][f'full_{mode}_embedding']))
        newD.append(np.dot(np.squeeze(enhanced_query), np.squeeze(all_input_data[i][f'{content_name}_embedding'])))
        newI.append(target_p_id)
        y_pred_list, true_list = merge_index(D, I, newD, newI)
        evaluate(true_list, y_pred_list, ndcg_score_story_all)
    print_performance(ndcg_score_story_all)
    return ndcg_score_story_all

def get_bm25_result(all_input_data, fpm,args=None,mode=None,cut_off=0, path_name=None):
    ndcg_score_story_all = get_result_template()

    id_name, content_name = mode2info[mode], mode
    index = dict(list([(key, transform2dict(item[f'full_{content_name}_content'].split())) for key, item in fpm.items()]))

    for i in tqdm.tqdm(range(len(all_input_data)), mininterval=300): # 1000
        query = all_input_data[i]['query_content']
        if len(query.split()) > args.query_length + cut_off:
            continue
        target_p_id = all_input_data[i][id_name]
        enhanced_query = transform2dict(query.split())
        y_pred_list = []
        true_list = []
        for k, v in index.items():
            if k == target_p_id:
                true_list.append(1)
                y_pred_list.append(BM25(enhanced_query, transform2dict(all_input_data[i][f'{content_name}_content'].split()), use_local_idf=False))
            else:
                true_list.append(0)
                y_pred_list.append(BM25(enhanced_query, v, use_local_idf=False))
        evaluate(true_list, y_pred_list, ndcg_score_story_all)
    print_performance(ndcg_score_story_all)
    return ndcg_score_story_all

def print_error(e):
    print(f'Error: {e}')

def get_bingxing(func, pool, *args, **args2):
    return pool.apply_async(func, args = (args,args2), error_callback = print_error)

def get_rm3_result(all_input_data, fpm,args=None,mode=None,cut_off=0,rm3_cut_off=3,path_name=None):
    ndcg_score_story_all = get_result_template()

    id_name, content_name = mode2info[mode], mode
    whoosh_index = built_index(fpm,mode=mode,path_name=path_name,reload=True)
    searcher = whoosh_index.searcher(weighting=scoring.BM25F())
    parser = QueryParser("content", whoosh_index.schema, group=OrGroup)
    fpm_index_woosh = dict(list([(str(key), transform2dict(item[f'full_{content_name}_content'].split())) for key, item in fpm.items()]))
    fpm_index = dict(list([(key, transform2dict(item[f'full_{content_name}_content'].split())) for key, item in fpm.items()]))

    for i in tqdm.tqdm(range(len(all_input_data)), mininterval=300): # 1000
        query = all_input_data[i]['query_content']
        if len(query.split()) > args.query_length + cut_off:
            continue
        target_p_id = all_input_data[i][id_name]
        enhanced_query = transform2dict(query.split())
        # first stage retrieval
        y_pred_list = []
        d_list = []
        for k, v in fpm_index.items():
            if k == target_p_id:
                d_list.append(transform2dict(all_input_data[i][f'{content_name}_content'].split()))
                y_pred_list.append(BM25(enhanced_query, d_list[-1], use_local_idf=False))
            else:
                d_list.append(v)
                y_pred_list.append(BM25(enhanced_query, v, use_local_idf=False))
        # second stage retrieval
        doc_estimate_list = sorted(list(zip(d_list, y_pred_list)), key=lambda v:v[1], reverse=True)
        estimate_list = [doc_estimate_list[i][1] for i in range(rm3_cut_off)]
        doc_list = [doc_estimate_list[i][0] for i in range(rm3_cut_off)]
        enhanced_query = rm3_expansion(enhanced_query, doc_list, estimate_list)
        enhanced_query = ' '.join(enhanced_query)
        y_pred_list, true_list = rank_with_my_woosh(searcher, parser, enhanced_query, fpm_index_woosh, target_p_id, target_document=transform2dict(all_input_data[i][f'{content_name}_content'].split()))
        evaluate(true_list, y_pred_list, ndcg_score_story_all)
    print_performance(ndcg_score_story_all)
    return ndcg_score_story_all
    
def entrophy_transform(all_input_data, args, original_query_content, output_data, key, entrophy):
    percentage = []
    for i in range(len(all_input_data)):
        all_input_data[i]['query_content'] = original_query_content[i]
    for i_tmp in range(len(all_input_data)):
        # 应该换成生成的置信度，而非目标的
        if args.sample == False:
            i = i_tmp
        else:
            i = i_tmp % len(output_data[key]['content_pred'])
        k = 0
        all_input_data[i]['content_pred'] = output_data[key]['content_pred'][i]
        content_pred_split = output_data[key]['content_pred'][i].split()
        if args.valid_loss:
            if output_data[key]['valid_loss'][i] < 2.0:
                all_input_data[i]['query_content'] = original_query_content[i] + ' ' + ' '.join(output_data[key]['content_pred'][i].split()[:args.cut_off])
                percentage.append(1)
            else:
                all_input_data[i]['query_content'] = original_query_content[i]
                percentage.append(0)
        elif args.entrophy:
            all_input_data[i]['query_content'] = original_query_content[i] 
            while k < len(entrophy['entrophy'][i]) and k < len(content_pred_split) and entrophy['entrophy'][i][k] < 1.2:
                all_input_data[i]['query_content'] += content_pred_split[k]
                k += 1
            percentage.append(1 if k > 0 else 0)
        else:
            all_input_data[i]['query_content'] = original_query_content[i] + ' ' + ' '.join(output_data[key]['content_pred'][i].split()[:args.cut_off])
            percentage.append(1)
    return percentage

if __name__ == '__main__':
    # load command args
    args = load_config()
    all_input_data, fpm, mode = load_data(args)
    
    if args.query_sample:
        if args.sample:
            all_input_data_list = []
            for k in range(10):
                all_input_data_raw = copy.deepcopy(all_input_data)
                for i in range(len(all_input_data_raw)):
                    all_input_data_raw[i]['query_content'] = ' '.join(random_selection(all_input_data_raw[i]['query_content'].split(), args.query_length))
                all_input_data_list.append(copy.deepcopy(all_input_data_raw))
            all_input_data = []
            for k in range(10):
                all_input_data += all_input_data_list[k]
        else:
            for i in range(len(all_input_data)):
                all_input_data[i]['query_content'] = ' '.join(random_selection(all_input_data[i]['query_content'].split(), args.query_length))
            # query_sample.append()
    
    if args.smaller_docs:
        id_name, content_name = mode2info[mode], mode
        target_p_ids = []
        for i in tqdm.tqdm(range(len(all_input_data)), mininterval=300): # 1000
            target_p_id = all_input_data[i][id_name]
            target_p_ids.append(target_p_id)
        fpm = dict(list([(key, item) for key, item in fpm.items() if key in target_p_ids]))
    
    # you should enable gpu service when running repllama
    if 'repllama' in args.model:
        result = {}
        repllam_result = {'query_embedding': [all_input_data[i]['query_embedding'] for i in range(len(all_input_data))]}
        result['original'] = get_repllama_result(repllam_result,all_input_data,fpm, args=args, mode=mode)
        write_down=f'results/{args.task_name}/repllama_pas_ql{args.query_length}.json'
    elif 'brainllm_rm3' in args.model:
        result = {}
        if args.all_random_number:
            output_data = {'brainllm':load_result_data(args, 'test_idf_0.5.json'), }
        else:
            output_data = {'brainllm':json.load(open(os.path.join(args.result_path, 'test_idf_0.5.json'))), }
        original_query_content = copy.deepcopy([all_input_data[i]['query_content'] for i in range(len(all_input_data))])
        for key in output_data.keys(): # 
            print(f"{key}: ")
            entrophy = load_entrophy(args, 'test_idf_0.5.entrophy.json') if args.entrophy else None
            percentage = entrophy_transform(all_input_data, args, original_query_content, output_data, key, entrophy)
            print(np.mean(percentage))
            result[key] = get_rm3_result(all_input_data,fpm, args=args, mode=mode,cut_off=args.cut_off,path_name=f'index/{args.task_name}_{args.common_data_prefix}')
        result['percentage'] = percentage
        if args.sample:
            write_down=f'results/{args.task_name}/{args.common_data_prefix}_combine_pas_ql{args.query_length}_sample_cf{args.cut_off}.json'
        else:
            write_down=f'results/{args.task_name}/{args.common_data_prefix}_combine_pas_ql{args.query_length}_cf{args.cut_off}.json'
    elif 'bm25' in args.model:
        bm25_func =  get_woosh_bm25_result # get_bm25_result
        result = {}
        print('original: ')
        result['original'] = bm25_func(all_input_data,fpm, args=args, mode=mode,path_name=f'index/{args.task_name}_{args.common_data_prefix}')
        if args.all_random_number:
            output_data = {'brainllm':load_result_data(args, 'test_idf_0.5.json'), 'perbrainllm':load_result_data(args, 'test_permutated_idf_0.5.json'), 'stdllm':load_result_data(args, 'test_nobrain_idf_0.5.json'),'brainllm-idf':load_result_data(args, 'test.json')}
        else:
            output_data = {'brainllm':json.load(open(os.path.join(args.result_path, 'test_idf_0.5.json'))), 'perbrainllm':json.load(open(os.path.join(args.result_path, 'test_permutated.json'))), 'stdllm':json.load(open(os.path.join(args.result_path, 'test_nobrain.json'))),'brainllm-idf':json.load(open(os.path.join(args.result_path, 'test.json')))}
        original_query_content = copy.deepcopy([all_input_data[i]['query_content'] for i in range(len(all_input_data))])
        for key in output_data.keys(): # 
            print(f"{key}: ")
            if args.entrophy:
                entrophy = load_entrophy(args, 'test_idf_0.5.entrophy.json')
            else:
                entrophy= None
            percentage = entrophy_transform(all_input_data, args, original_query_content, output_data, key, entrophy)
            print(np.mean(percentage))
            result[key] = bm25_func(all_input_data,fpm, args=args, mode=mode,cut_off=args.cut_off,path_name=f'index/{args.task_name}_{args.common_data_prefix}')
        for i_tmp in range(len(all_input_data)):
            if args.sample == False:
                i = i_tmp
            else:
                i = i_tmp % len(output_data[key]['content_pred'])
            all_input_data[i]['query_content'] = original_query_content[i] + ' ' + ' '.join(output_data[key]['content_true'][i].split()[:args.cut_off])            
        print(f"perceived: ")
        result['perceived'] = bm25_func(all_input_data,fpm, args=args, mode=mode,cut_off=args.cut_off,path_name=f'index/{args.task_name}_{args.common_data_prefix}')
        result['percentage'] = percentage
        middle_fix = '_sample' if args.sample else ''
        middle_fix = middle_fix + '_all' if args.all_random_number else middle_fix
        write_down=f'results/{args.task_name}/{args.common_data_prefix}_bm25_pas_ql{args.query_length}{middle_fix}_cf{args.cut_off}.json'
        print(write_down)
    elif 'rm3' in args.model:
        result = {}
        result['rm3'] = get_rm3_result(all_input_data,fpm, args=args, mode=mode,path_name=f'index/{args.task_name}_{args.common_data_prefix}')
        write_down=f'results/{args.task_name}/{args.common_data_prefix}_rm3_pas_ql{args.query_length}_cf{args.cut_off}.json'
    elif 'repllama_rank' in args.model:
        brain_llm_embedding = json.load(open(os.path.join(args.result_path, 'test.rank.json')))
        perbrain_llm_embedding = json.load(open(os.path.join(args.result_path, 'test_permutated.rank.json')))
        llm_embedding = json.load(open(os.path.join(args.result_path, 'test_nobrain.rank.json')))
        result = {}
        result['brainllm'] = get_repllama_ranking_result(brain_llm_embedding,all_input_data,fpm, args=args,mode=mode)
        result['perbrainllm'] = get_repllama_ranking_result(perbrain_llm_embedding,all_input_data, fpm, args=args,mode=mode)
        result['stdllm'] = get_repllama_ranking_result(llm_embedding,all_input_data, fpm,args=args,mode=mode)
        write_down=f'results/{args.task_name}/repllama_rank_pas_ql{args.query_length}.json'
    elif 'repllama_train' in args.model:
        result = {}
        repllam_result = {'query_embedding': [all_input_data[i]['query_embedding'] for i in range(len(all_input_data))]}
        result['original'] = get_repllama_result(repllam_result,all_input_data,fpm, args=args, mode=mode)
        write_down=f'results/{args.task_name}/repllama_train_pas_ql{args.query_length}.json'
    elif 'repllama_rank_train' in args.model:
        brain_llm_embedding = json.load(open(os.path.join(args.result_path, 'test.train.rank.json')))
        result = {}
        result['brainllm'] = get_repllama_ranking_result(brain_llm_embedding,all_input_data,fpm, args=args,mode=mode)
        write_down=f'results/{args.task_name}/repllama_rank_train_pas_ql{args.query_length}.json'
    # save the results
    os.makedirs(f'results/{args.task_name}', exist_ok=True)
    json.dump(result, open(write_down, 'w'))

    