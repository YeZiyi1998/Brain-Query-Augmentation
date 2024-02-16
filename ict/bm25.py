import numpy as np
import math
import json
import copy
import os
from collections import Counter

# from system.utils import softmax

def transform2dict(arr1):
    return Counter(arr1)

def softmax(logits):
	e_x = np.exp(logits)
	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
	return probs

rm3_paras = {'avg_doc_len':58.7, 'k1':1.2, 'k3':1.2, 'b':0.75, 'lambda':0.6, 'add_count':10}

w2idf = json.load(open(''))
w2idx = json.load(open(''))
    
local_idf = None
def load_local_idf(data_path):
    global local_idf
    if os.path.exists(f'data/w2idf_{data_path}.json'):
        local_idf = json.load(open(f'data/w2idf_{data_path}.json'))
    else:
        print('failure to load local idf')
    return local_idf

def load_local_cf(data_path):
    if os.path.exists(f'data/w2idf_{data_path}.json'):
        local_cf = json.load(open(f'data/w2cf_{data_path}.json'))
    else:
        print('failure to load local idf')
    return local_cf

def BM25(q_dic, d_dic,use_local_idf=True):
    global rm3_paras, w2idf, local_idf
    doc_len = np.sum([item for item in d_dic.values()])
    bm25_score = 0
    for w in q_dic.keys():
        if w in d_dic.keys():
            sq = (rm3_paras['k3'] + 1) * q_dic[w] / (rm3_paras['k3'] + q_dic[w])
            K = rm3_paras['k1'] * (1 - rm3_paras['b'] + rm3_paras['b'] * doc_len / rm3_paras['avg_doc_len'])
            sd = (rm3_paras['k1'] + 1) * d_dic[w] / (K + d_dic[w])
            if use_local_idf:
                idf = local_idf[w] #if w in local_idf.keys() else 10
            else:
                idf = w2idf[str(w2idx[w])] if w in w2idx.keys() and str(w2idx[w]) in w2idf.keys() else 10
            bm25_score += sd * sq * idf
    return bm25_score

def LM(q_dic, d_dic):
    global rm3_paras, w2idf, local_idf
    lm_score = 0
    for w in q_dic.keys():
        if w in d_dic.keys():
            lm_score += d_dic[w] * w2idf[w]
    return lm_score

def rm3_expansion(q_dic, d_dic_list, estimate_list):
    global rm3_paras, w2idf, local_idf
    def add2dic(re_dic, w, v):
        if w in re_dic.keys():
            re_dic[w] += v
        else:
            re_dic[w] = v
    # get add words with current docs
    word2rel = {}
        
    estimate_list = softmax(estimate_list)
    for j in range(0, len(d_dic_list)):
        pm = estimate_list[j]
        doc_len = np.sum([item for item in d_dic_list[j].values()])
        pq = 1
        for w in q_dic.keys():
            if w in d_dic_list[j].keys():
                pqi = rm3_paras['lambda'] * d_dic_list[j][w] / doc_len + (1 - rm3_paras['lambda']) / math.exp(w2idf[str(w2idx[w])] if w in w2idx.keys() and str(w2idx[w]) in w2idf.keys() else 10)
            else:
                pqi = (1 - rm3_paras['lambda']) / math.exp(w2idf[str(w2idx[w])] if w in w2idx.keys() and str(w2idx[w]) in w2idf.keys() else 10)
            pq *= pqi
        for w in d_dic_list[j].keys():
            pwm = rm3_paras['lambda'] * d_dic_list[j][w] / doc_len + (1 - rm3_paras['lambda']) / math.exp(w2idf[str(w2idx[w])] if w in w2idx.keys() and str(w2idx[w]) in w2idf.keys() else 10)            
            add2dic(word2rel, w, pq * pm * pwm)
    word2rel_sorted = sorted(word2rel.items(), key = lambda v: v[1], reverse = True)
    new_q_p = copy.deepcopy(q_dic)
    current_count = 0
    for w_ in word2rel_sorted:
        w = w_[0]
        if current_count == rm3_paras['add_count']:
            break
        if w not in new_q_p:
            new_q_p[w] = 1
            current_count += 1
    return new_q_p
    