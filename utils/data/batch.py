
import numpy as np
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse import eye as sp_eye
from scipy.sparse import tril as sp_tril

from .common.helpers import *

from collections import defaultdict

import os
from math import floor



class ClickBatchGenerator(object):

    def __init__(self, rv_map, clicks):

        self.clicks = clicks
        self.subgraph_aggr = SubGraphAggregator(rv_map, self.clicks)

    def gen(self, batch_size, agg_layers, tril=True, self_attn=None, seed=None, yield_eog=True):

        np.random.seed(seed)

        random_click_indexes = np.random.permutation(len(self.clicks))

        # 以click edge中的query與doc計算相似度並訓練模型，所以batch中的單位為click edge
        for batch_head in range(0, len(random_click_indexes), batch_size):

            # 抓取相鄰的nodes
            batch_click_indexes = random_click_indexes[batch_head:batch_head+batch_size]
            rec_ids =  [rec_id for click_index in batch_click_indexes for rec_id in self.clicks[click_index][:2]]

            subgraph_seqs, subgraph_adj, subgraph_features = self.subgraph_aggr.aggregate(rec_ids, agg_layers, tril, self_attn)

            q_seqs, rel_rec_seqs = subgraph_seqs.reshape(-1,2).transpose((1,0))

            subgraph_adj_coo = subgraph_adj.tocoo()
            adj_indices = np.vstack([subgraph_adj_coo.row, subgraph_adj_coo.col]).transpose((1,0))

            subgraph_features_coo = subgraph_features.tocoo()
            feature_indices = np.vstack([subgraph_features_coo.row, subgraph_features_coo.col]).transpose((1,0))

            batch = {
                'q_seqs': q_seqs,
                'rel_rec_seqs': rel_rec_seqs,
                'adj_indices': adj_indices,
                'adj_data': subgraph_adj.data,
                'adj_shape': subgraph_adj.shape,
                'feature_indices': feature_indices,
                'feature_data': subgraph_features.data,
                'feature_shape': subgraph_features.shape,
            }

            if yield_eog:
                yield batch, ((batch_head+batch_size)>=len(self.clicks))
            else:
                yield batch


class RecordBatchGenerator(object):

    def __init__(self, rv_map, clicks, rec_ids=None, subgraph_aggr=None):

        self.subgraph_aggr = SubGraphAggregator(rv_map, clicks) if not isinstance(subgraph_aggr, SubGraphAggregator) else subgraph_aggr

        self.set_records(rec_ids)

    def gen(self, batch_size, agg_layers, tril=True, self_attn=None, log_scale=False, yield_eog=True):

        assert (self.rec_ids is not None), 'Records not set.'

        for batch_head in range(0, len(self.rec_ids), batch_size):

            batch_rec_ids = self.rec_ids[batch_head:batch_head+batch_size]

            subgraph_seqs, subgraph_adj, subgraph_features = self.subgraph_aggr.aggregate(batch_rec_ids, agg_layers, tril, self_attn, log_scale)

            subgraph_adj_coo = subgraph_adj.tocoo()
            adj_indices = np.vstack([subgraph_adj_coo.row, subgraph_adj_coo.col]).transpose((1,0))

            subgraph_features_coo = subgraph_features.tocoo()
            feature_indices = np.vstack([subgraph_features_coo.row, subgraph_features_coo.col]).transpose((1,0))

            batch = {
                'rec_seqs': subgraph_seqs,
                'adj_indices': adj_indices,
                'adj_data': subgraph_adj.data,
                'adj_shape': subgraph_adj.shape,
                'feature_indices': feature_indices,
                'feature_data': subgraph_features.data,
                'feature_shape': subgraph_features.shape,
            }

            if yield_eog:
                yield batch, ((batch_head+batch_size)>=len(self.rec_ids))
            else:
                yield batch

    def set_records(self, rec_ids):

        self.rec_ids = list(rec_ids) if class_name(rec_ids)=='generator' else rec_ids


class RankBatchGenerator(object):

    def __init__(self, rv_map, clicks, sample_weight_types=None):

        self.subgraph_aggr = SubGraphAggregator(rv_map, clicks)

        rec_seq_map = self.subgraph_aggr.rec_seq_map
        query_nbrs_map = defaultdict(list)

        assert sample_weight_types is None or len(sample_weight_types)==len(clicks[0][3:]), '`sample_weight_types` doesn\'t matched with corresponding parts in `clicks`'
        self.sample_weight_types = sample_weight_types

        for click_pair in clicks:
            if click_pair[0] in rec_seq_map and click_pair[1] in rec_seq_map:
                nbr = click_pair[1:] if self.sample_weight_types else click_pair[1:3]
                query_nbrs_map[click_pair[0]].append(nbr)

        self.query_nbrs_pairs = list(query_nbrs_map.items())
        self.rank_num = len(self.query_nbrs_pairs)

    def gen(self, batch_size, rank_len, agg_layers, tril=True, self_attn=None, log_scale=False, rand_rank_selector='uniform', rank_pad_seq=-1, seed=None, ord_seed=None):

        np.random.seed(ord_seed if ord_seed else seed)
        # 讓random_pair_indexes的seed能與其他部份的seed分開，方便產出sub_repos時每個sub_repo能採用相同的query順序(便於切分val_set)
        random_pair_indexes = np.random.permutation(len(self.query_nbrs_pairs))

        np.random.seed(seed)

        for batch_head in range(0, len(random_pair_indexes), batch_size):

            q_ids, rel_doc_ids = [], []
            nbr_lens= []
            if self.sample_weight_types:
                sample_weights = {weight_type:[] for weight_type in self.sample_weight_types}

            for pair_index in random_pair_indexes[batch_head:batch_head+batch_size]:

                q_id, nbrs = self.query_nbrs_pairs[pair_index]

                if len(nbrs)>rank_len:

                    if rand_rank_selector=='click':
                        probs = [nbr[1] for nbr in nbrs]
                        nbrs = [nbrs[i] for i in np.random.choice(len(nbrs), rank_len, replace=False, p=probs)]

                    else:
                        nbrs = np.random.permutation(nbrs)[:rank_len]

                if self.sample_weight_types is None:
                    nbr_ids = [nbr[0] for nbr in sorted(nbrs, key=lambda x:x[1], reverse=True)]

                else:
                    nbr_ids, clicks, *nbr_weights = list(zip(*sorted(nbrs, key=lambda x:x[1], reverse=True)))
                    for weight_type, weights_by_type in zip(self.sample_weight_types, nbr_weights):
                        sample_weights[weight_type].append(weights_by_type)

                q_ids.append(q_id)
                rel_doc_ids.extend(nbr_ids)
                nbr_lens.append(len(nbr_ids))

            rec_ids = q_ids + rel_doc_ids
            subgraph_seqs, subgraph_adj, subgraph_features = self.subgraph_aggr.aggregate(rec_ids, agg_layers, tril, self_attn, log_scale)

            q_seqs = subgraph_seqs[:len(q_ids)]

            rel_rec_seqs = []
            seq_cursor = len(q_ids)
            for nbr_len in nbr_lens:
                seq_cursor_next = seq_cursor + nbr_len
                rel_rec_seqs.append(subgraph_seqs[seq_cursor:seq_cursor_next])
                seq_cursor = seq_cursor_next
            rel_rec_seqs = tail_padding(rel_rec_seqs, rank_pad_seq, rank_len)

            if self.sample_weight_types:
                for weight_type in self.sample_weight_types:
                        sample_weights[weight_type] = tail_padding(sample_weights[weight_type], 0, rank_len)

            subgraph_adj_coo = subgraph_adj.tocoo()
            adj_indices = np.vstack([subgraph_adj_coo.row, subgraph_adj_coo.col]).transpose((1,0))

            subgraph_features_coo = subgraph_features.tocoo()
            feature_indices = np.vstack([subgraph_features_coo.row, subgraph_features_coo.col]).transpose((1,0))

            batch = {
                'q_seqs': q_seqs,
                'rel_rec_seqs': rel_rec_seqs,
                'adj_indices': adj_indices,
                'adj_data': subgraph_adj.data,
                'adj_shape': subgraph_adj.shape,
                'feature_indices': feature_indices,
                'feature_data': subgraph_features.data,
                'feature_shape': subgraph_features.shape,
            }

            if self.sample_weight_types:
                for weight_type, weights_by_type in sample_weights.items():
                    batch['sample_weights_%s'%weight_type] = weights_by_type

            yield batch, ((batch_head+batch_size)>=len(random_pair_indexes))



class SubGraphAggregator(object):

    def __init__(self, rv_map, edges):

        self.rv_map = rv_map
        self.rec_seq_map = self.rv_map.rec_seq_map

        self.adj = self._init_adj(edges)

    def _init_adj(self, edges):

        rows, cols, data = [], [], []

        for edge in edges:

            rec_id_a, rec_id_b, weight = edge[:3]

            if rec_id_a not in self.rec_seq_map or rec_id_b not in self.rec_seq_map:
                continue

            rec_seq_a, rec_seq_b = self.rec_seq_map[rec_id_a], self.rec_seq_map[rec_id_b]

            rows.extend([rec_seq_a, rec_seq_b])
            cols.extend([rec_seq_b, rec_seq_a])
            data.extend([weight, weight])

        shape = (len(self.rec_seq_map), len(self.rec_seq_map))

        adj = csr_matrix((data, (rows, cols)), shape)

        return adj

    def aggregate(self, rec_ids, agg_layers, tril=True, self_attn=None, log_scale=False):

        agged_nodes = set()  # 已聚合鄰接關係的節點
        bottom_nodes = set(self.rec_seq_map[rec_id] for rec_id in rec_ids)  # 最底層、尚未聚合鄰接關係的節點
        for __ in range(agg_layers):

            nbrs = set(self.adj[list(bottom_nodes)].indices)
            agged_nodes.update(bottom_nodes)

            bottom_nodes = nbrs - agged_nodes

            if len(bottom_nodes)==0:
                break

        yield_seqs = list(agged_nodes) + list(bottom_nodes)

        # 將源自rec_ids的yield_seqs由原本rec_id的座標空間映射到subgraph_seq的座標空間
        subgraph_seq_map = {yield_seq: subgraph_seq for subgraph_seq, yield_seq in enumerate(yield_seqs)}
        subgraph_seqs = np.array([subgraph_seq_map[self.rec_seq_map[rec_id]] for rec_id in rec_ids])

        # 只要node和edge都在subgraph就納入adj，不排除最後一層bottom_nodes的edges
        subgraph_adj = self.adj[yield_seqs][:,yield_seqs]

        if tril:
            subgraph_adj = sp_tril(subgraph_adj, k=-1)
        if isinstance(self_attn, int):
            subgraph_adj += sp_eye(len(yield_seqs), dtype=np.int32) * self_attn
        elif self_attn=='max_click':
            subgraph_adj += dia_matrix((subgraph_adj.max(axis=1).data, [0]), subgraph_adj.shape)
        if log_scale:
            subgraph_adj.data = np.log(subgraph_adj.data)

        subgraph_features = self.rv_map.get_by_seqs(yield_seqs)

        return subgraph_seqs, subgraph_adj, subgraph_features


def iter_block_batches(block_repo_path, block_type, sub_repo_seq=None, block_range=None, batch_limit=None,
                       get_adj=True, adj_type='normal', self_attn=1, sample_weight_type=None, seed=None):

    if batch_limit is None:
        batch_limit = float('inf')

    if sub_repo_seq is not None and os.path.exists(block_repo_path+'/sub_repo.0'):
        sub_repo_seq %= len(os.listdir(block_repo_path))
        block_repo_path = relpath('sub_repo.%s'%sub_repo_seq, block_repo_path)

    block_list = np.array(os.listdir(block_repo_path))

    if isinstance(block_range, tuple) and len(block_range)==2:
        block_list = block_list[list(range(*block_range))]
    elif block_range is not None:
        block_list = block_list[block_range]

    if seed is not None:
        np.random.seed(seed)
        block_list_seqs = np.random.permutation(len(block_list))
    else:
        # 若順序不需打亂，則按照block_seqs排序(若block_type='search'，因可能需與另外的rec_ids配合而要依照順序)
        block_list_seqs = [pair[0] for pair in sorted([(seq, int(block_name.split('.')[-1])) for seq, block_name in enumerate(block_list)],
                                                      key=lambda x:x[1])]

    block_list = block_list[block_list_seqs]

    print('Iterating batches from %s blocks of %s'%(len(block_list), block_repo_path), flush=True)

    batch_count = 0
    for block_name in block_list:

        block_path = relpath(block_name, block_repo_path)
        block_seq = block_name.split('.')[-1]
        block = {
            'seqs': np.load(relpath('seqs.%s.npy'%block_seq, block_path), allow_pickle=True),
            'shapes': np.load(relpath('shapes.%s.npy'%block_seq, block_path)),
            # 'adjs_f': np.load(relpath('adjs.%s.npy'%i, block_path)),
            'features_f': np.load(relpath('features.%s.npy'%block_seq, block_path)),
        }
        if get_adj: block['adjs_f'] = np.load(relpath('adjs.%s.npy'%block_seq, block_path))  # f for flattened
        if sample_weight_type:
            block['sample_weights'] = np.load(relpath('sample_weights_%s.%s.npy'%(sample_weight_type, block_seq), block_path))

        adjs_cursor, features_cursor = 0, 0
        for batch_seq in range(len(block['seqs'])):

            seqs, shapes = block['seqs'][batch_seq], block['shapes'][batch_seq]
            if sample_weight_type: sample_weights = block['sample_weights'][batch_seq]

            batch_count += 1
            if batch_count > batch_limit:
                return

            adj_shape, feature_shape, (adj_len, feature_len) = shapes

            features = block['features_f'][features_cursor:features_cursor+feature_len].reshape(3,-1)
            features_cursor += feature_len

            if get_adj:

                feature_indices, feature_data = features[:2].transpose((1,0)), features[2]

                adj = block['adjs_f'][adjs_cursor:adjs_cursor+adj_len].reshape(3,-1)

                adjs_cursor += adj_len

                # 處理indices

                nbr_indices = adj[:2].transpose((1,0))
                # 因目前未區分兩點間的in/out degree，而batch_adj中只儲存其中一個方向的weight，因此在此將另一方向的edge weight也填上
                nbr_indices = np.vstack([nbr_indices, np.flip(nbr_indices, axis=1)])
                self_attn_indices = np.repeat(np.arange(adj_shape[0]), 2, axis=0).reshape(-1,2)  # 讓自身也能夠參考到自己

                adj_indices = np.vstack([nbr_indices, self_attn_indices])

                # 處理data

                if adj_type=='equal':  # all equal to one

                    adj_data = np.array([1]*len(adj[2])*2+[1]*(adj_shape[0]))

                else:

                    nbr_data = np.log(adj[2]) if adj_type=='log_scale' else adj[2]
                    nbr_data = np.concatenate([nbr_data, nbr_data])
                    self_attns = (csr_matrix((nbr_data, nbr_indices.transpose((1,0))), adj_shape).max(axis=1).toarray().reshape(-1) if self_attn=='max_click' else
                                  [np.log(self_attn) if adj_type=='log_scale' else self_attn]*adj_shape[0])

                    adj_data = np.concatenate([nbr_data, self_attns])

            else:

                # 若不需adj的話，只需yield records的feature

                features_mat = csr_matrix((features[2], features[:2]), feature_shape)
                orig_seqs_shape = seqs.shape
                all_rec_seqs = seqs.reshape(-1)

                try:
                    rec_features_coo = features_mat[all_rec_seqs].tocoo()
                except:
                    # seqs偶爾會是像[array(), array()]的ndarray(通常都為[[],[]]，也許跟batch不同有關?)，也因此先前需要設定allow_pickle
                    all_rec_seqs = np.concatenate(seqs)
                    rec_features_coo = features_mat[all_rec_seqs].tocoo()

                feature_indices = np.vstack([rec_features_coo.row, rec_features_coo.col]).transpose((1,0))
                feature_data, feature_shape = rec_features_coo.data, rec_features_coo.shape

                seqs = np.arange(len(all_rec_seqs))
                # 已於rec_features_coo處篩選出rec_seqs的features(篩掉其他只存在於adj的rec_seqs的features)，因此seqs改採用篩選時rec順序
                if block_type=='train':
                    seqs = seqs.reshape(*orig_seqs_shape)


            batch = {
                'feature_indices': feature_indices,
                'feature_data': feature_data,
                'feature_shape': feature_shape,
            }

            if block_type=='train':
                batch.update({'q_seqs': seqs[0], 'rel_rec_seqs': seqs[1]} if len(seqs)==2 else  # for click batch
                             {'q_seqs': seqs[:,0], 'rel_rec_seqs': seqs[:,1:]})                 # for rank/hop batch
            else:
                batch.update({'rec_seqs':seqs})

            if get_adj:
                batch.update({
                    'adj_indices': adj_indices,
                    'adj_data': adj_data,
                    'adj_shape': adj_shape,
                })

            if sample_weight_type:
                batch['sample_weights'] = sample_weights

            yield batch


def iter_merged_block_batches(block_repo_path, block_type, block_range=None, batch_limit=None, merge_batches=1, seed=None):

    if batch_limit is None:
        batch_limit = float('inf')

    queue = {
        'seqs': [],
        'feature_indices': [],
        'feature_data': [],
    }

    batch_count = 0
    queued_rec_num = 0

    for batch_seq, batch in enumerate(iter_block_batches(block_repo_path, block_type, block_range, get_adj=False, seed=seed)):

        if block_type=='train':
            queue['seqs'].append([batch['q_seqs']+queued_rec_num, batch['rel_rec_seqs']+queued_rec_num])

        else:
            queue['seqs'].append([batch['rec_seqs']+queued_rec_num])

        queue['feature_indices'].append(batch['feature_indices'] + [queued_rec_num, 0])
        queue['feature_data'].append(batch['feature_data'])

        queued_rec_num += batch['feature_shape'][0]

        if (batch_seq+1)%merge_batches==0:

            yield __merge_queued_batches(block_type, queue, (queued_rec_num, batch['feature_shape'][1]))

            for attr in queue:
                queue[attr] = []
            queued_rec_num = 0

            batch_count += 1
            if batch_count >= batch_limit:
                return

    if len(queue['seqs'])>0:
        yield __merge_queued_batches(block_type, queue, (queued_rec_num, batch['feature_shape'][1]))

def __merge_queued_batches(block_type, queue, feature_shape):

    queue_seqs = np.hstack(queue['seqs'])

    batch = {
        'feature_indices': np.vstack(queue['feature_indices']),
        'feature_data': np.concatenate(queue['feature_data']).reshape(-1),
        'feature_shape': feature_shape
    }

    batch.update({'q_seqs': queue_seqs[0], 'rel_rec_seqs': queue_seqs[1]} if block_type=='train' else {'rec_seqs':queue_seqs[0]})

    return batch
