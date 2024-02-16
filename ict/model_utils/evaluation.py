from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np

def recall(y_true, y_pred, k):
    sorted_y = sorted(zip(y_true, y_pred), key=lambda v: v[1], reverse=True)
    sorted_y_true = [item[0] for item in sorted_y]
    return float(np.sum(np.array(sorted_y_true)[:k]) > 0)       
      
def evaluate(y_true, y_pred, save_dic):
    for k in [1,3,5,10,20,100]:
        save_dic[f'ndcg@{k}'].append(ndcg_score([y_true], [y_pred], k=k))
    for k in [20,100]:
        save_dic[f'recall@{k}'].append(recall(y_true, y_pred, k=k))
    save_dic['y_pred'].append([float(item) for item in y_pred])
    save_dic['y_true'].append([float(item) for item in y_true])

def mrr_map_evaluate(y_true_list, y_pred_list, save_dic):
    save_dic['map'] = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        save_dic['map'].append(average_precision_score(y_true, y_pred))
   
def print_performance(save_dic, name=None):
    if name is not None:
        print('------'+name+'------')
    for k in [1,3,5,10,20,100]:
        print(f'ndcg@{k}: ', f"{'{:.3f}'.format(np.mean(save_dic[f'ndcg@{k}']))}+{'{:.3f}'.format(np.std(save_dic[f'ndcg@{k}']))}", end = '\t')
    for k in [20,100]:
        print(f'recall@{k}: ', f"{'{:.3f}'.format(np.mean(save_dic[f'recall@{k}']))}+{'{:.3f}'.format(np.std(save_dic[f'recall@{k}']))}", end = '\t')
    if 'map' in save_dic.keys():
        print(f'map: ', f"{'{:.3f}'.format(np.mean(save_dic[f'map']))}+{'{:.3f}'.format(np.std(save_dic[f'map']))}", end = '\t')
    print()
    
def get_result_template():
    return {'ndcg@10':[], 'query':[], 'doc':[],'query_add':[], 'ndcg@1':[], 'ndcg@3':[], 'ndcg@5':[], 'recall@100':[], 'recall@20':[],'y_pred':[],'y_true':[],'ndcg@20':[],'ndcg@100':[],}
