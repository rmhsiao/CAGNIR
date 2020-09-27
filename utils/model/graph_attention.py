
import numpy as np
import tensorflow as tf

from .layers import Layer
from .inits import glorot, zeros, ones


# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/PetarV-/GAT
# which is under an identical MIT license as CAGNIR



class GraphAttention(Layer):

    def __init__(self, input_dim, hid_units, n_heads, act,
                 attn_drop=0.0, ft_drop=0.0, batch_norm=False, residual=False, **kwargs):

        super(GraphAttention, self).__init__(**kwargs)

        self._hid_units = hid_units
        self._n_heads = n_heads

        self._attn_drop = attn_drop
        self._ft_drop = ft_drop

        self._act = act

        self._batch_norm = batch_norm
        self._residual = residual

        self._log_tfvars = kwargs['log_tfvars']

        self._get_attn_id = lambda agg_seq, head_seq: '%s-%s'%(agg_seq, head_seq)

        self._init_vars(input_dim, hid_units, n_heads, self._residual)

    def _call(self, inputs, **kwargs):

        features = inputs
        adj_mat = kwargs['adj_mat']

        node_num = tf.shape(adj_mat)[0]

        for agg_seq, output_dim in enumerate(self._hid_units):

            with tf.variable_scope('agg_layer_%s'%agg_seq):

                # if (agg_seq+1)!=len(self._hid_units):
                act, residual = self._act, self._residual
                attn_output_dim = int(output_dim/self._n_heads[agg_seq])
                # else:
                #     act, residual = lambda x: x, False
                #     attn_output_dim = output_dim

                attns = []
                for head_seq in range(self._n_heads[agg_seq]):

                    attn_id = self._get_attn_id(agg_seq, head_seq)

                    attns.append(self.sp_attn_head(features, output_dim=attn_output_dim, node_num=node_num, adj_mat=adj_mat, act=act,
                                                   attn_id=attn_id, ft_drop=self._ft_drop, coef_drop=self._attn_drop,
                                                   residual=residual))

                # if (agg_seq+1)!=len(self._hid_units):
                features = tf.concat(attns, axis=-1)
                if self._batch_norm:
                    features = tf.layers.batch_normalization(features, training=kwargs['bn_training'])
                # else:
                #     logits = tf.add_n(attns) / len(attns)

        # return logits
        return features

    def sp_attn_head(self, features, output_dim, node_num, adj_mat, act, attn_id,
                     ft_drop, coef_drop, residual):

        with tf.variable_scope('attn_%s'%attn_id):

            if ft_drop != 0.0:
                features = tf.nn.dropout(features, rate=ft_drop)

            feature_trans = tf.matmul(features, self._tfvars['weights_trans_%s'%attn_id], name='feature_trans')

            with tf.variable_scope('click_attns'):
                click_attns = tf.SparseTensor(indices=adj_mat.indices,
                                              values=tf.identity(adj_mat.values, name='click_attns'),
                                              dense_shape=adj_mat.dense_shape)

            with tf.variable_scope('sim_attns'):

                feature_trans_norm = tf.nn.l2_normalize(feature_trans, axis=1, name='ft_norm')

                ft_x_by_adj = tf.nn.embedding_lookup(feature_trans_norm, adj_mat.indices[:,0], name='sim_attns_x')
                ft_y_by_adj = tf.nn.embedding_lookup(feature_trans_norm, adj_mat.indices[:,1], name='sim_attns_y')

                sim_attns = tf.reduce_sum(tf.multiply(ft_x_by_adj, ft_y_by_adj), axis=1, name='cos_sim')
                sim_attns = tf.SparseTensor(indices=adj_mat.indices,
                                            values=tf.identity(sim_attns, name='sim_attns'),
                                            dense_shape=adj_mat.dense_shape)

            with tf.variable_scope('nn_attns'):

                # https://github.com/PetarV-/GAT/issues/15
                ft_x = tf.matmul(feature_trans, self._tfvars['weights_ftx_%s'%attn_id], name='nn_attns_ftx')  # node x
                ft_y = tf.matmul(feature_trans, self._tfvars['weights_fty_%s'%attn_id], name='nn_attns_fty')  # node y

                unweighted_adj_mat = tf.SparseTensor(adj_mat.indices, tf.ones_like(adj_mat.values, name='unweighted_adj'), adj_mat.dense_shape)

                # 依據x、y座標選取特徵，亦即將節點特徵分別指定至adj上的所屬座標位置(如ft_x_by_adj中，單一row內皆為對應seq的ft_x[seq]，數量為其len(nbrs))
                # (此處不若dense的版本可以直接相乘，因adj中絕大多數座標皆為空，直接相乘將消耗過多記憶體)
                ft_x_by_adj = unweighted_adj_mat * ft_x
                ft_y_by_adj = unweighted_adj_mat * tf.transpose(ft_y, [1,0])

                logits = tf.sparse_add(ft_x_by_adj, ft_y_by_adj)
                nn_attns = tf.SparseTensor(indices=logits.indices,
                                           values=tf.nn.leaky_relu(logits.values),
                                           dense_shape=logits.dense_shape)
                nn_attns = tf.sparse_softmax(nn_attns, name='nn_attns')

            if self._log_tfvars:
                tf.summary.histogram('attn_%s_clicks'%attn_id, click_attns.values)
                tf.summary.histogram('attn_%s_sims'%attn_id, sim_attns.values)
                tf.summary.histogram('attn_%s_nn'%attn_id, nn_attns.values)

            # ---

            feature_trans = tf.reshape(feature_trans, (-1,output_dim))

            view_names = ['self', 'click', 'sim', 'nn']

            with tf.variable_scope('view_reprs'):

                view_reprs_list = [feature_trans]

                gamma_funcs = [lambda x:x]*3

                for attns, gamma_func, view_name in zip([click_attns, sim_attns, nn_attns], gamma_funcs, view_names[1:]):

                    attns = tf.SparseTensor(indices=attns.indices, values=gamma_func(attns.values), dense_shape=attns.dense_shape)
                    view_reprs_list.append(tf.sparse_tensor_dense_matmul(tf.sparse_softmax(attns),
                                                                         feature_trans,
                                                                         name='%s_view_reprs'%view_name))

            with tf.variable_scope('view_attns'):

                view_attn_list = []
                for view_reprs, view_name in zip(view_reprs_list, view_names):

                    view_attns = tf.matmul(tf.concat([feature_trans, view_reprs], axis=1), self._tfvars['weights_view_%s'%attn_id])
                    view_attn_list.append(tf.nn.leaky_relu(tf.reshape(view_attns, (tf.shape(features)[0],)), name='%s_view_attns'%view_name))

                view_coefs = tf.expand_dims(tf.nn.softmax(tf.exp(tf.stack(view_attn_list, axis=1)), axis=1), axis=-1, name='all_view_coefs')

                if self._log_tfvars:
                    for i, view_name in zip(range(len(view_reprs_list)), view_names):
                        tf.summary.histogram('attn_%s_view_%s'%(attn_id, view_name), view_coefs[:,i])

            with tf.variable_scope('head_reprs'):

                output = tf.identity(sum([(view_reprs * view_coefs[:,i]) for i, view_reprs in enumerate(view_reprs_list)]) + self._tfvars['bias_%s'%attn_id],
                                     name='aggr_reprs')


                # residual connection
                if residual:
                    if features.shape[-1] != output.shape[-1]:
                        output += tf.matmul(features, self._tfvars['weights_res_%s'%attn_id]) + self._tfvars['bias_res_%s'%attn_id]
                    else:
                        output += features

                return tf.identity(act(output), name='head_reprs_%s'%attn_id)  # activation

    def _init_vars(self, input_dim, hid_units, n_heads, residual):

        for agg_seq, input_dim in enumerate([input_dim] + self._hid_units[:-1]):

            # output_dim = int(self._hid_units[agg_seq]/self._n_heads[agg_seq]) if (agg_seq+1)!=len(self._hid_units) else self._hid_units[agg_seq]
            output_dim = int(self._hid_units[agg_seq]/self._n_heads[agg_seq])

            for head_seq in range(self._n_heads[agg_seq]):
                attn_id = self._get_attn_id(agg_seq, head_seq)
                with tf.variable_scope('attn_vars_%s'%attn_id):

                    self._tfvars['weights_trans_%s'%attn_id] = glorot([input_dim, output_dim], get_var=True, name='weights_trans')

                    self._tfvars['weights_ftx_%s'%attn_id] = glorot([output_dim, 1], get_var=True, name='weights_ftx')
                    self._tfvars['weights_fty_%s'%attn_id] = glorot([output_dim, 1], get_var=True, name='weights_fty')

                    self._tfvars['weights_view_%s'%attn_id] = glorot([output_dim*2, 1], get_var=True, name='weights_view')

                    self._tfvars['bias_%s'%attn_id] = zeros([output_dim], get_var=True, name='bias')

                    # if residual and (agg_seq+1)!=len(self._hid_units):
                    if residual:
                        self._tfvars['weights_res_%s'%attn_id] = glorot([input_dim, output_dim], get_var=True, name='weights_res')
                        self._tfvars['bias_res_%s'%attn_id] = zeros([output_dim], get_var=True, name='bias_res')
