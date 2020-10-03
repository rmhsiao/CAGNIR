
import numpy as np

from math import log



def ndcg(search_ranks, groundtruth_labels, top_k=10, ver='1', get_mean=True):

    calc_gain = (lambda rel: rel) if ver.startswith('1') else (lambda rel: (2**rel-1))

    ndcgs = []

    for q_id, groundtruth in groundtruth_labels.items():

        if groundtruth['idcg'] is None or len(groundtruth['idcg'])==0:
            continue

        if q_id not in search_ranks:
            ndcgs.append(np.zeros(top_k))
            continue

        gain = [calc_gain(groundtruth['docs'].get(doc_id, 0.)) for doc_id, doc_rel in search_ranks[q_id][:top_k]]

        dcg = [gain[0]]
        for i in range(1, top_k):
            dcg.append(gain[i]/(log(i+2, 2)) + dcg[i-1])

        ndcg = np.array(dcg) / np.array(groundtruth['idcg'])[:top_k]

        ndcgs.append(ndcg)

    ndcgs = np.array(ndcgs)

    return ndcgs.mean(axis=0) if get_mean else ndcgs
