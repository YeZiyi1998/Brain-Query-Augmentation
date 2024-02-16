from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
from model_utils.evaluation import evaluate, print_performance, get_result_template
from part1_config import load_config, mode2info
import tqdm   
import os 
from bm25 import BM25, transform2dict
import numpy as np
def test():
    schema = Schema(title=TEXT(stored=True), content=TEXT)
    ix = create_in("index", schema)
    writer = ix.writer()
    writer.add_document(title=u"First document", 
                    content=u"This is the first document we've added!")
    writer.add_document(title=u"Second document", 
                    content=u"The second one is even more interesting!")
    writer.commit()
    
    searcher = ix.searcher(weighting=scoring.BM25F())
    query = QueryParser("content", ix.schema).parse("first")
    results = searcher.search(query, limit=100)
    print(str(results))
    print(results[0])

def built_index(fpm,mode=None,path_name='',reload=False):
    if os.path.exists(path_name) == False:
        os.mkdir(path_name)
    elif reload == False:
        schema = Schema(title=TEXT(stored=True), content=TEXT)
        return create_in(path_name, schema)
    else:
        os.system(f'rm {path_name}/*')
    
    id_name, content_name = mode2info[mode], mode
    index = dict(list([(key, item[f'full_{content_name}_content']) for key, item in fpm.items()]))

    schema = Schema(title=TEXT(stored=True), content=TEXT)
    ix = create_in(path_name, schema)
    writer = ix.writer()
    
    print('indexing documents ....')
    
    for k,v in index.items():
        writer.add_document(title=str(k), content=v)
    writer.commit()
    return ix

def whoosh_bm25(idf, tf, fl, avgfl, B, K1):
    # idf - inverse document frequency
    # tf - term frequency in the current document
    # fl - field length in the current document
    # avgfl - average field length across documents in collection
    # B, K1 - free paramters
    return idf * ((tf * (K1 + 1)) / (tf + K1 * ((1 - B) + B * fl / avgfl)))

def rank_with_my_woosh(searcher, parser, enhanced_query_raw: str, fpm_index, target_p_id, target_document):
    enhanced_query = parser.parse(enhanced_query_raw)
    enhanced_query_raw = transform2dict(enhanced_query_raw.split())
    results = searcher.search(enhanced_query, limit=120)
    y_pred_list = []
    true_list = []
    for item in results:
        k = item['title']
        if k != str(target_p_id):
            y_pred_list.append(BM25(enhanced_query_raw, fpm_index[k], use_local_idf=False))
            true_list.append(0)
    for _ in range(120-len(results)):
        y_pred_list.append(0)
        true_list.append(0)
    y_pred_list.append(BM25(enhanced_query_raw, target_document, use_local_idf=False))
    true_list.append(1)
    return y_pred_list, true_list 

def get_woosh_bm25_result(all_input_data, fpm, args=None, mode=None, cut_off=0,path_name=None):
    ndcg_score_story_all = get_result_template()
    id_name, content_name = mode2info[mode], mode
    
    whoosh_index = built_index(fpm,mode=mode,path_name=path_name,reload=True)
    searcher = whoosh_index.searcher(weighting=scoring.BM25F())
    parser = QueryParser("content", whoosh_index.schema, group=OrGroup)
    fpm_index = dict(list([(str(key), transform2dict(item[f'full_{content_name}_content'].split())) for key, item in fpm.items()]))
    
    for i in tqdm.tqdm(range(len(all_input_data)), mininterval=300): # 1000
        query = all_input_data[i]['query_content']
        if len(query.split()) > args.query_length + cut_off:
            continue
        target_p_id = all_input_data[i][id_name]
        enhanced_query_raw = query
        target_document = transform2dict(all_input_data[i][f'{content_name}_content'].split())
        y_pred_list, true_list = rank_with_my_woosh(searcher, parser, enhanced_query_raw, fpm_index, target_p_id, target_document)
        evaluate(true_list, y_pred_list, ndcg_score_story_all)
    print_performance(ndcg_score_story_all)
    return ndcg_score_story_all
    
if __name__ == '__main__':
    test()


