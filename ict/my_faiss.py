import faiss
import numpy as np
import bisect

def build_index(doc_vectors, ids, dimension=4096):
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(np.array(doc_vectors), np.array(ids))
    return index

def retrieve_from_index(index, q, top_k=100):
    q = np.array(q).reshape(1,-1)
    D, I = index.search(q, top_k)
    return D[0], I[0]

def merge_index(D, I, target_scores,target_ids):
    D, I = [D[idx] for idx in range(len(D)) if I[idx] not in target_ids], [0 for idx in range(len(D)) if I[idx] not in target_ids]
    for i in range(len(target_ids)):
        insert_point = len(D) - bisect.bisect(D[::-1], target_scores[i])
        D.insert(insert_point, target_scores[i])
        I.insert(insert_point, 1)
    return D, I
