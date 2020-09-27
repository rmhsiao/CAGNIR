
import numpy as np
import tensorflow as tf
# from tensorflow.python.client import timeline

from utils.environ import env
from utils.helpers import *
from utils.mlogging import mlogging

from utils.model_config import ModelConfig
from utils.layers import Dense, build_mlp
from utils.word_embeddings import WordHashing
from utils.graph_attention import GraphAttention
from utils.vsm import vsm_search

from dataclasses import dataclass
from functools import partial
from math import exp
import os



@dataclass
class CAGNIRConfig(ModelConfig):

    input_size: int
    agg_layers: int
    loss_type: str
    gamma: float
    irrel_num: int

    learning_rate: float
    hid_units: list
    n_heads: list

    batch_norm: bool
    residual: bool
    weight_decay: float

    rank_pad_seq: int = -1
    use_sample_weights: bool = False
    seed: int = 18035

class CAGNIR(object):

    def __init__(self, model_id, load_model=False, model_config=None, logger=None, log_tfvars=False):

        self.model_id = model_id

        if load_model:  # 載入先前模型
            self._model_save_path = relpath('%s/%s'%(class_name(self).lower(), self.model_id), env('MODELBASE_DIR'))

        else:  # 建立新模型
            self._model_save_path = None

            self._use_sample_weights = model_config.use_sample_weights

            tf.set_random_seed(model_config.seed)

        self._logger = logger if logger is not None else mlogging.get_logger(prefix=class_name(self))

        self._log_tfvars = log_tfvars

        self._tfops = {}
        self._placeholders = {}

        if self._model_save_path is None:
            self._build(*model_config.astuple())


    def _build(self, input_size, agg_layers, loss_type, gamma, irrel_num,
               learning_rate, hid_units, n_heads, batch_norm, residual, weight_decay, rank_pad_seq=-1, use_sample_weights=False, *args):

        with tf.variable_scope('inputs'):

            self._placeholders = {
                'rec_seqs': tf.placeholder(tf.int32, (None,), name='rec_seqs'),
                'rel_rec_seqs': tf.placeholder(tf.int32, ((None,) if loss_type=='dssm_loss' else (None,None)), name='rel_rec_seqs'),
                'adj': tf.sparse.reorder(tf.sparse_placeholder(tf.float32, name='adj')),
                'features': tf.sparse.reorder(tf.sparse_placeholder(tf.float32, name='features')),

                'attn_drop': tf.placeholder_with_default(0., shape=(), name='attn_drop'),
                'ft_drop': tf.placeholder_with_default(0., shape=(), name='ft_drop'),
                'bn_training': tf.placeholder_with_default(False, shape=(), name='bn_training'),
            }
            if use_sample_weights and loss_type!='dssm_loss':
                self._placeholders['sample_weights'] = tf.placeholder(tf.float32, (None,None), name='sample_weights')

        hid_cursor = 0  # hid_units_cursor

        with tf.variable_scope('word_hashing'):

            word_hashing_layer = WordHashing(input_size, hid_units[hid_cursor], batch_norm=batch_norm, log_tfvars=self._log_tfvars)
            self._tfops['word_hashings'] = word_hashing_layer(self._placeholders['features'], bn_training=self._placeholders['bn_training'])

            hid_cursor += 1

            # dense_layer = Dense(hid_units[hid_cursor-1], hid_units[hid_cursor], act=tf.nn.relu, batch_norm=batch_norm, name='word_hashings_trans', log_tfvars=self._log_tfvars)
            # self._tfops['word_hashings'] = dense_layer(self._tfops['word_hashings'], bn_training=self._placeholders['bn_training'])

            # hid_cursor += 1

        with tf.variable_scope('graph_attentions'):

            graph_attention_layers = GraphAttention(hid_units[hid_cursor-1], hid_units[hid_cursor:hid_cursor+agg_layers], n_heads, tf.nn.elu,
                                                    self._placeholders['attn_drop'], self._placeholders['ft_drop'], batch_norm, residual,
                                                    log_tfvars=self._log_tfvars)

            self._tfops['graph_aggs'] = graph_attention_layers(self._tfops['word_hashings'], adj_mat=self._placeholders['adj'], bn_training=self._placeholders['bn_training'])

        with tf.variable_scope('feature_trans'):

            self._tfops['feature_trans'] = build_mlp(self._tfops['word_hashings'], hid_units[hid_cursor:hid_cursor+agg_layers],
                                                     batch_norm, self._placeholders['bn_training'], self._log_tfvars, 'feature_trans', last_act=True)

        hid_cursor += agg_layers

        with tf.variable_scope('reprs'):

            self._tfops['collab_reprs'] = tf.concat([self._tfops['graph_aggs'], self._tfops['feature_trans']], axis=1, name='collab_reprs')

            if len(hid_units) > hid_cursor:

                with tf.variable_scope('collab_repr_trans'):

                    self._tfops['collab_reprs'] = build_mlp(self._tfops['collab_reprs'], hid_units[hid_cursor:],
                                                            batch_norm, self._placeholders['bn_training'], self._log_tfvars, 'collab_repr_trans', last_act=False)

            self._tfops['reprs'] = tf.nn.l2_normalize(self._tfops['collab_reprs'], axis=1)

        with tf.variable_scope('repr_vectors'):

            self._tfops['rec_reprs'] = tf.nn.embedding_lookup(self._tfops['reprs'], self._placeholders['rec_seqs'], name='rec_reprs')

            cand_rec_seqs = self._get_cand_rec_seqs(self._placeholders['rel_rec_seqs'], loss_type, irrel_num, rank_pad_seq)
            self._tfops['cand_rec_reprs'] = tf.nn.embedding_lookup(self._tfops['reprs'], cand_rec_seqs, name='cand_rec_reprs')

        with tf.variable_scope('relevances'):

            self._tfops['relevances'] = tf.reduce_sum(tf.multiply(tf.expand_dims(self._tfops['rec_reprs'], axis=1), self._tfops['cand_rec_reprs']), axis=2, name='relevances_op')
                                        # shape: (batch_num, rank_len|irrel_num)
            if self._log_tfvars:

                rank_len = tf.shape(self._placeholders['rel_rec_seqs'])[1]
                self._tfops['rel_summaries'] = tf.tuple([
                        tf.summary.histogram('relevances', self._tfops['relevances']),
                        tf.summary.histogram('relevances_pos', self._tfops['relevances'][:,0]),
                        tf.summary.histogram('relevances_pos_', self._tfops['relevances'][:,1:rank_len]),
                        tf.summary.histogram('relevances_neg', self._tfops['relevances'][:,rank_len:]),
                    ], name='rel_summaries')

        self._build_train_ops(self._tfops['relevances'], loss_type, gamma, irrel_num, learning_rate, weight_decay,
                              self._placeholders['rec_seqs'], self._placeholders['rel_rec_seqs'], self._placeholders['adj'],
                              (self._placeholders['sample_weights'] if use_sample_weights else None), rank_pad_seq)


    def _build_train_ops(self, relevances, loss_type, gamma, irrel_num, learning_rate, weight_decay,
                         rec_seqs=None, rel_rec_seqs=None, adj=None, sample_weights=None, rank_pad_seq=None):

        with tf.variable_scope('metrics'):

            # gamma = tf.Variable(gamma, dtype=env('TF_FLOATX'), name='gamma_var')
            # if self._log_tfvars:
            #     tf.summary.scalar('gamma', gamma)

            with tf.variable_scope('loss_ops'):

                if loss_type=='attention_rank':

                    if sample_weights is not None:
                        rel_rec_seqs_shape = tf.shape(rel_rec_seqs)
                        cand_rec_weights = tf.concat([sample_weights, tf.zeros((rel_rec_seqs_shape[0], irrel_num-rel_rec_seqs_shape[1]))], axis=1)

                    else:
                        cand_rec_weights = self._get_cand_rec_weights(rec_seqs, rel_rec_seqs, adj, tf.shape(relevances), rank_pad_seq)

                    # a_y = tf.nn.softmax(tf.exp(cand_rec_weights*10), axis=1)
                    a_y = tf.nn.softmax(cand_rec_weights, axis=1)
                    a_s = tf.nn.softmax(relevances, axis=1)

                    losses = -1 * (a_y * tf.log(a_s) + (1 - a_y) * tf.log(1 - a_s))

                elif loss_type=='pair_loss':

                    distances = 1 - (tf.expand_dims(relevances[:,0], 1) - relevances[:,1:])
                    losses = tf.where(distances<0, tf.zeros_like(distances), distances)

                else:
                    losses = -tf.log(tf.nn.softmax(relevances*gamma, axis=1)[:,0])

                if weight_decay!=0.:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * weight_decay

                else:
                    l2_loss = 0.

                self._tfops['loss'] = tf.identity(tf.reduce_mean(losses) + l2_loss, name='loss')

            with tf.variable_scope('raw_soft_max'):
                raw_soft_max = tf.nn.softmax(relevances, axis=1, name='raw_soft_max')

            with tf.variable_scope('raw_loss_ops'):

                raw_losses = -tf.log(raw_soft_max[:,0])
                self._tfops['raw_loss'] = tf.reduce_mean(raw_losses, name='raw_loss')

            with tf.variable_scope('raw_rel_ratio_ops'):
                self._tfops['raw_rel_ratio'] = tf.identity(tf.math.exp(self._tfops['raw_loss']*-1) * 100, name='raw_rel_ratio')

            with tf.variable_scope('acc_ops'):
                correct_count = tf.count_nonzero(tf.equal(tf.argsort(raw_soft_max, axis=1)[:,-1], 0))
                self._tfops['acc'] = tf.identity((correct_count / tf.cast(tf.shape(raw_soft_max)[0], tf.int64)) * 100, 'acc')

            self._tfops['metrics'] = tf.tuple([self._tfops[metric] for metric in ['loss', 'raw_loss', 'raw_rel_ratio', 'acc']], name='metric_ops')

            if self._log_tfvars:
                for metric in ['loss', 'raw_loss', 'raw_rel_ratio', 'acc']:
                    tf.summary.scalar(metric, self._tfops[metric])

        with tf.variable_scope('train'):

            self._tfops['train'] = tf.group([
                tf.train.AdamOptimizer(learning_rate).minimize(self._tfops['loss']),
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch normalization
            ], name='train')


    def _get_cand_rec_seqs(self, rel_rec_seqs, loss_type, irrel_num, rank_pad_seq=None):

        cond = lambda rel_rec_seqs, irrel_rec_seqs: tf.reduce_any(tf.equal(rel_rec_seqs, irrel_rec_seqs))
        body = lambda rel_rec_seqs, irrel_rec_seqs: (rel_rec_seqs, tf.random_shuffle(rel_rec_seqs))

        with tf.variable_scope('get_irrel_recs'):

            irrel_rec_seqs_list = []
            for i in range(irrel_num):
                if loss_type=='dssm_loss':
                    __, irrel_rec_seqs = tf.while_loop(cond, body, [rel_rec_seqs, rel_rec_seqs], name='irrel_recs_%s'%i)
                else:
                    __, irrel_rec_seqs = tf.while_loop(cond, body, [rel_rec_seqs[:,0], rel_rec_seqs[:,0]], name='irrel_recs_%s'%i)

                irrel_rec_seqs_list.append(irrel_rec_seqs)

        if loss_type=='dssm_loss':
            return tf.transpose(tf.stack([rel_rec_seqs]+irrel_rec_seqs_list, axis=0))  # (len(rel_rec_seqs), 1+irrel_num)

        else:

            irrel_rec_seqs = tf.stack(irrel_rec_seqs_list, axis=1)

            cand_rec_seqs = tf.where(tf.not_equal(rel_rec_seqs, rank_pad_seq), rel_rec_seqs, irrel_rec_seqs[:,:tf.shape(rel_rec_seqs)[1]])
            cand_rec_seqs = tf.concat([cand_rec_seqs, irrel_rec_seqs[:,tf.shape(rel_rec_seqs)[1]:]], axis=1)

            return cand_rec_seqs

    def _get_cand_rec_weights(self, rec_seqs, rel_rec_seqs, adj, cands_shape, rank_pad_seq):

        non_pad_coords = tf.where(tf.not_equal(rel_rec_seqs, rank_pad_seq))
        q_coords = tf.gather(rec_seqs, non_pad_coords[:,0])
        rel_rec_coords = tf.cast(tf.stack([q_coords, tf.gather_nd(rel_rec_seqs, non_pad_coords)], axis=1), tf.int64)

        adj_indices_seqs_of_rel_recs = tf.where(tf.reduce_all(tf.equal(adj.indices, tf.expand_dims(rel_rec_coords, axis=1)), axis=-1))[:,1]
        rel_rec_weights = tf.gather(adj.values, adj_indices_seqs_of_rel_recs)

        cand_rec_weights = tf.sparse.to_dense(tf.SparseTensor(non_pad_coords, rel_rec_weights, tf.cast(cands_shape, tf.int64)), default_value=1e-12)
        # cand_rec_weights = tf.sparse.to_dense(tf.SparseTensor(non_pad_coords, tf.exp(rel_rec_weights), tf.cast(cands_shape, tf.int64)), default_value=1e-12)

        return cand_rec_weights


    def train(self, batch_gen, val_batch_gen, epochs, attn_drop=None, ft_drop=None, bn_training=False, log_freqs=[], max_step=None, early_stop_threshold=0.1,
              gpu_limit_frac=0.95, report_upon_oom=False):

        log_types = ['train', 'val', 'val_info', 'summary', 'init_summary', 'reg_ckpt']
        log_freqs = {log_type: freq for log_type, freq in zip(log_types, list(log_freqs))}

        if max_step is None:
            max_step = float('inf')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_limit_frac)
        config = tf.ConfigProto(gpu_options=gpu_options)

        with tf.Session(config=config) as s:

            # 初始化session相關處理與設定

            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom=report_upon_oom)
            # run_metadata = tf.RunMetadata()
            # s_run = partial(s.run, options=run_options, run_metadata=run_metadata)
            s_run = partial(s.run, options=tf.RunOptions(report_tensor_allocations_upon_oom=report_upon_oom))

            if self._model_save_path is None:
                saver = tf.train.Saver(save_relative_paths=True, max_to_keep=None)
                reg_saver = tf.train.Saver(save_relative_paths=True, keep_checkpoint_every_n_hours=1, max_to_keep=None)
                s_run(tf.global_variables_initializer())

            else:
                saver = self._restore_model(s)  # include restore tfops & placeholders
                reg_saver = tf.train.Saver(save_relative_paths=True, keep_checkpoint_every_n_hours=1, max_to_keep=None)

            ckpt_save_path_proto = relpath('%s/%s/%s.%%s.ckpt'%(class_name(self).lower(), self.model_id, class_name(self)), env('MODELBASE_DIR'))
            reg_ckpt_save_path = relpath('%s/reg_ckpts/%s/%s.ckpt'%(class_name(self).lower(), self.model_id, class_name(self)), env('MODELBASE_DIR'))

            if self._log_tfvars:
                summaries = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(relpath('%s/tensorboard/%s/train'%(class_name(self).lower(), self.model_id), env('LOG_DIR')), graph=s.graph)
                val_summary_writer = tf.summary.FileWriter(relpath('%s/tensorboard/%s/val'%(class_name(self).lower(), self.model_id), env('LOG_DIR')))


            self._logger.info('Network initialized / restored.')

            step = 0
            val_step = 0
            min_loss = {'loss':float('inf'), 'raw_loss':float('inf')}
            none_op = tf.no_op()
            for epoch in range(epochs):

                self._logger.info('Epoch #%s\n%s'%(epoch, '#'*12))

                for batch_seq, batch in enumerate(batch_gen(epoch)):

                    step += 1
                    batch_seq += 1

                    feed_dict = {
                        self._placeholders['rec_seqs']: batch['q_seqs'],
                        self._placeholders['rel_rec_seqs']: batch['rel_rec_seqs'],
                        self._placeholders['adj']: (batch['adj_indices'], batch['adj_data'], batch['adj_shape']),
                        self._placeholders['features']: (batch['feature_indices'], batch['feature_data'], batch['feature_shape']),
                    }
                    if attn_drop is not None: feed_dict[self._placeholders['attn_drop']] = attn_drop
                    if ft_drop is not None: feed_dict[self._placeholders['ft_drop']] = ft_drop
                    if bn_training: feed_dict[self._placeholders['bn_training']] = True
                    if self._use_sample_weights: feed_dict[self._placeholders['sample_weights']] = batch['sample_weights']

                    to_log_for_train = step%log_freqs['train']==0 or step%log_freqs['reg_ckpt']==0
                    to_log_for_summary = self._log_tfvars and (step%log_freqs['summary']==0 or (step<log_freqs['summary'] and step%log_freqs['init_summary']==0))

                    run_ops = [self._tfops['train'], (self._tfops['metrics'] if to_log_for_train else none_op)]  # none_op只是為了使summaries固定為results[2]
                    if  to_log_for_summary:
                        run_ops.append(summaries)

                    results = s_run(run_ops, feed_dict=feed_dict)

                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open(relpath('timeline/timeline.%s.json'%self.model_id, env('LOG_DIR')), 'w') as f:
                    #     f.write(ctf)
                    #     exit()

                    if to_log_for_summary:
                        # summary_writer.add_run_metadata(run_metadata, 'step%s'%step)
                        summary_writer.add_summary(results[2], step)

                    if to_log_for_train:
                        self._logger.info('Epoch#%s, batch#%s, step#%s - loss: %.4f, raw_loss: %.4f, raw_rel_ratio: %.2f%%, acc: %.2f%%'%(
                                            epoch, batch_seq, step, *results[1]))

                    if step%log_freqs['reg_ckpt']==0:
                        reg_saver.save(s, reg_ckpt_save_path, global_step=step)
                        self._logger.info('Regular checkpoint saved.')


                    if step%log_freqs['val']==0:

                        val_step += 1

                        *val_metrics, rel_summaries = self._validate(s_run, val_batch_gen, val_step, log_freqs['val_info'])

                        val_info = 'Epoch#%s, batch#%s, step#%s (val#%s) - loss: %.4f, raw_loss: %.4f, raw_rel_ratio: %.2f%%, acc: %.2f%%'%(
                                       epoch, batch_seq, step, val_step, *val_metrics)

                        # 根據不同的metric儲存模型
                        save_types = []
                        for i, metric_name in enumerate(['loss', 'raw_loss']):
                            if val_metrics[i]<min_loss[metric_name]:
                                min_loss[metric_name] = val_metrics[i]
                                save_types.append(metric_name)

                        if len(save_types)>0:
                            save_info =  '(%s)'%(','.join(save_types))
                            val_info += ' - model saved %s'%save_info
                            saver.save(s, ckpt_save_path_proto%save_info, global_step=step)

                        self._logger.info(val_info)

                        if self._log_tfvars:

                            metric_tags = ['metrics/%s'%metric_name for metric_name in ['loss', 'raw_loss', 'raw_rel_ratio', 'acc']]

                            for tag, val_metric in zip(metric_tags, val_metrics):

                                val_summary_writer.add_summary(tf.Summary(value=[
                                    tf.Summary.Value(tag=tag, simple_value=val_metric),
                                ]), step)

                            for rel_summary in rel_summaries:
                                val_summary_writer.add_summary(rel_summary, step)

                        if val_metrics[1]>(min_loss['raw_loss']+early_stop_threshold):
                            self._logger.info('Earlily stopped.')
                            return

                    if step>=max_step:
                        return

    def _validate(self, s_run, val_batch_gen, val_step, val_info_log_freq):

        batch_metrics_sums  = [0]*4
        rel_summaries = None

        for batch_seq, batch in enumerate(val_batch_gen(0)):

            feed_dict = {
                self._placeholders['rec_seqs']: batch['q_seqs'],
                self._placeholders['rel_rec_seqs']: batch['rel_rec_seqs'],
                self._placeholders['adj']: (batch['adj_indices'], batch['adj_data'], batch['adj_shape']),
                self._placeholders['features']: (batch['feature_indices'], batch['feature_data'], batch['feature_shape']),
            }
            if self._use_sample_weights: feed_dict[self._placeholders['sample_weights']] = batch['sample_weights']

            if self._log_tfvars and batch_seq==0:
                batch_metrics, rel_summaries = s_run([self._tfops['metrics'], self._tfops['rel_summaries']], feed_dict=feed_dict)

            else:
                batch_metrics = s_run(self._tfops['metrics'], feed_dict=feed_dict)

            if batch_seq%val_info_log_freq==0:
                self._logger.info('val_batch#%s (val#%s) - loss: %.4f, raw_loss: %.4f, raw_rel_ratio: %.2f%%, acc: %.2f%%'%(
                                    batch_seq, val_step, *batch_metrics))  # 'loss', 'raw_loss', 'raw_rel_ratio', 'acc'

            for i in (0,1,3):
                batch_metrics_sums[i] += batch_metrics[i]

        loss = batch_metrics_sums[0] / (batch_seq+1)

        raw_loss = batch_metrics_sums[1] / (batch_seq+1)
        raw_rel_ratio = exp(raw_loss*-1) * 100

        acc = batch_metrics_sums[3] / (batch_seq+1)

        return loss, raw_loss, raw_rel_ratio, acc, rel_summaries


    def search(self, test_queries_batch_gen, docs_batch_gen, test_query_ids, doc_ids, rel_calc_freq, top_k=10, ckpt_step=None, repr_type='cagnir'):

        with tf.Session() as s:

            # 初始化session相關處理與設定

            s_run = partial(s.run)

            saver = self._restore_model(s, ckpt_step, repr_type)  # include restore tfops & placeholders

            self._logger.info('Network restored.')

            test_query_reprs = np.vstack(list(self._run_rec_reprs(s_run, test_queries_batch_gen)))
            doc_reprs_gen = self._run_rec_reprs(s_run, docs_batch_gen)

            search_ranks = vsm_search(test_query_reprs, doc_reprs_gen, test_query_ids, doc_ids, rel_calc_freq, top_k, logger=self._logger)

            return search_ranks

    def _run_rec_reprs(self, s_run, batch_gen):

        for batch in batch_gen():

            feed_dict = {
                self._placeholders['rec_seqs']: batch['rec_seqs'],
                self._placeholders['adj']: (batch['adj_indices'], batch['adj_data'], batch['adj_shape']),
                self._placeholders['features']: (batch['feature_indices'], batch['feature_data'], batch['feature_shape']),
            }

            rec_reprs = s_run(self._tfops['rec_reprs'], feed_dict=feed_dict)

            yield rec_reprs


    def _restore_model(self, s, ckpt_step=None, repr_type='cagnir'):

        if ckpt_step is None:

            ckpt_path = tf.train.latest_checkpoint(self._model_save_path)

        else:

            ckpt_path = None

            for fname in os.listdir(self._model_save_path):

                comps = fname.split('.')[:-1]

                if comps and ckpt_step==comps[-1].split('-')[-1]:
                    ckpt_path = relpath('.'.join(comps), self._model_save_path)
                    break

        self._logger.info('Loading model from %s.'%ckpt_path)

        saver = tf.train.import_meta_graph('%s.meta'%ckpt_path)
        saver.restore(s, ckpt_path)

        placeholder_names = [
            'rec_seqs', 'rel_rec_seqs',
            'adj/indices', 'adj/values', 'adj/shape',
            'features/indices', 'features/values', 'features/shape',
            'attn_drop', 'ft_drop', 'bn_training',
        ]
        for ph_name in placeholder_names:
            self._placeholders[ph_name] = s.graph.get_tensor_by_name('inputs/%s:0'%ph_name)

        self._placeholders['adj'] = tf.SparseTensor(self._placeholders['adj/indices'], self._placeholders['adj/values'], self._placeholders['adj/shape'])
        self._placeholders['features'] = tf.SparseTensor(self._placeholders['features/indices'], self._placeholders['features/values'], self._placeholders['features/shape'])

        if repr_type=='dssm':
            reprs = s.graph.get_tensor_by_name('feature_trans/feature_trans.1/add:0')
            self._tfops['rec_reprs'] = tf.nn.embedding_lookup(reprs, self._placeholders['rec_seqs'], name='rec_reprs')

        else:
            self._tfops['rec_reprs'] = s.graph.get_tensor_by_name('repr_vectors/rec_reprs:0')

        return saver
