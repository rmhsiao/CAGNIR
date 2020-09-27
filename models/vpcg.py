
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np

from utils.environ import env
from utils.helpers import *
from utils.mlogging import mlogging

import os
import json



class VPCG(object):

    def __init__(self, model_id, logger=None):

        self.model_id = model_id

        self._logger = logger if logger else mlogging.get_logger(prefix=class_name(self))


    def train(self, query_data, doc_data, clicks, semantic_space, keep_features, batch_size, max_iter=None, epsilon=1e-12, self_aggr=False, log_freq=None):

        if type(max_iter)!=int:
            max_iter = float('inf')
        # if type(log_freq)!=int:
        #     log_freq = float('inf')

        model_save_path = relpath('vpcg/%s'%self.model_id, env('MODELBASE_DIR'))
        os.makedirs(model_save_path, exist_ok=True)

        (query_reprs, q_ids), (doc_reprs, doc_ids) = query_data, doc_data

        rec_ids = (rec_id for rec_ids in (q_ids, doc_ids) for rec_id in rec_ids)
        rec_seq_map = {rec_id: rec_seq for rec_seq, rec_id in enumerate(rec_ids)}

        with open(relpath('rec_seq_map.json', model_save_path), 'w') as f:
            json.dump(rec_seq_map, f)

        adj = self._init_adj(clicks, rec_seq_map)

        if semantic_space=='query':
            aggr_side = 1  #先從另一側開始聚合
            query_reprs = sparse_l2norm_by_row(query_reprs, epsilon, batch_size)
            doc_reprs = csr_matrix((len(doc_ids), query_reprs.shape[1]))

        else:
            aggr_side = 0
            doc_reprs = sparse_l2norm_by_row(doc_reprs, epsilon, batch_size)
            query_reprs = csr_matrix((len(q_ids), doc_reprs.shape[1]))

        side_adjs = [adj[:len(q_ids)][:,len(q_ids):], adj[len(q_ids):][:,:len(q_ids)]]
        # side_max_clicks = [side_adj.max(axis=1).toarray().reshape(-1,1) for side_adj in side_adjs]
        side_reprs = [query_reprs, doc_reprs]

        self._logger.info('Training data prepared.')

        n_iter = 0
        while True:

            self._logger.info('Iter #%s training'%n_iter)

            distances = []

            for __ in range(2):

                aggr_num = side_adjs[aggr_side].shape[0]
                log_num = int(aggr_num / batch_size / log_freq) if type(log_freq)==int else float('inf')
                aggr_side_name = 'query' if aggr_side==0 else 'doc'

                new_side_reprs_queue = []

                for batch_seq, batch_head in enumerate(range(0, aggr_num, batch_size)):

                    batch_tail = batch_head + batch_size

                    rec_reprs_piece = side_reprs[aggr_side][batch_head:batch_tail]
                    new_rec_reprs_piece = side_adjs[aggr_side][batch_head:batch_tail] * side_reprs[(aggr_side+1)%2]  # 從另一側聚合資訊
                    if self_aggr:
                        new_rec_reprs_piece += rec_reprs_piece  # rec_reprs_piece.multiply(side_max_clicks[aggr_side][batch_head:batch_tail])

                    new_rec_reprs_piece = sparse_top_k_by_row(sparse_l2norm_by_row(new_rec_reprs_piece, epsilon), keep_features)
                    new_side_reprs_queue.append(new_rec_reprs_piece)

                    distances.extend(sparse.linalg.norm((new_rec_reprs_piece - rec_reprs_piece), axis=-1))

                    if batch_seq%log_num==0 or (batch_tail+1)>=aggr_num:
                        self._logger.info('Iter #%s (%s), batch #%s (%.2f%%) processed'%(n_iter, aggr_side_name, batch_seq, (min(batch_tail+1, aggr_num)/aggr_num)*100))

                side_reprs[aggr_side] = sparse.vstack(new_side_reprs_queue)
                # 完成此次iter在aggr_side的聚合，將聚合完的資訊更新至side_reprs以利另一側的聚合

                aggr_side = (aggr_side + 1) % 2

            distances = np.array(distances)
            self._logger.info('Reprs of iter #%s trained, distances: %.6f (sum), %.6f (mean)'%(n_iter, distances.sum(), distances.mean()))

            all_rec_reprs = sparse.vstack(side_reprs)
            sparse.save_npz(relpath('%s.iter_%s.npz'%(self.model_id, n_iter), model_save_path), all_rec_reprs, compressed=True)

            self._logger.info('Reprs of iter #%s saved'%n_iter)

            n_iter += 1
            if n_iter>=max_iter:
                break

    def _init_adj(self, clicks, rec_seq_map):

        rows, cols, data = [], [], []

        missed_click_count = 0
        for rec_id_a, rec_id_b, click in clicks:

            if rec_id_a not in rec_seq_map or rec_id_b not in rec_seq_map:
                missed_click_count += 1
                continue

            rec_seq_a, rec_seq_b = rec_seq_map[rec_id_a], rec_seq_map[rec_id_b]

            rows.extend([rec_seq_a, rec_seq_b])
            cols.extend([rec_seq_b, rec_seq_a])
            data.extend([click, click])

        shape = (len(rec_seq_map), len(rec_seq_map))

        adj = csr_matrix((data, (rows, cols)), shape)

        if missed_click_count>0:
            self._logger.info('Missed clicks num: %s'%missed_click_count)

        return adj
