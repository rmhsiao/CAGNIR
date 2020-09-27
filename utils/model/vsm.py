
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict



def vsm_search(test_query_reprs, doc_reprs_gen, test_query_ids, doc_ids, rel_calc_freq, top_k=10, is_sparse_input=False, logger=None):

    list_vstack = sparse.vstack if is_sparse_input else np.vstack

    # test_query_reprs = list_vstack(list(test_queries_vecs_gen))

    if logger:
        logger.info('Test queries processed.')

    tmp_search_rels = defaultdict(lambda:[])
    batch_doc_reprs_queue = []

    doc_count = 0
    for batch_seq, batch_doc_reprs in enumerate(doc_reprs_gen):

        doc_count += batch_doc_reprs.shape[0]
        batch_doc_reprs_queue.append(batch_doc_reprs)

        if (batch_seq+1)%rel_calc_freq==0 or doc_count>=len(doc_ids):

            # 累積至一定數量後就先計算相關程度並擷取top_k，避免儲存過多結果而使記憶體爆掉

            queue_doc_reprs = list_vstack(batch_doc_reprs_queue)
            doc_seq_head = doc_count - queue_doc_reprs.shape[0]
            batch_doc_reprs_queue = []

            queue_rels = cosine_similarity(test_query_reprs, queue_doc_reprs)  # np.ndarray

            for tq_seq, top_queue_doc_seqs in enumerate(np.argsort(queue_rels, axis=1)[:,-top_k:]):

                tmp_search_rels[tq_seq] += list(zip(top_queue_doc_seqs+doc_seq_head, queue_rels[tq_seq][top_queue_doc_seqs].astype(np.float64)))
                # rels需轉為float64，否則無法以json儲存

            if logger:
                logger.info('%s (%.2f%%) query-doc relevances calculation processed.'%(doc_count, (doc_count/len(doc_ids)*100)))


    search_ranks = {}
    for tq_seq, rels in tmp_search_rels.items():

        top_rels = sorted(rels, key=lambda doc_rel: doc_rel[1], reverse=True)[:top_k]
        search_ranks[test_query_ids[tq_seq]] = [(doc_ids[doc_seq], rel) for doc_seq, rel in top_rels]

    return search_ranks
