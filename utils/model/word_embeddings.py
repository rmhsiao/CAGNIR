
import tensorflow as tf

from .layers import Layer
from .inits import zeros, glorot



class WordHashing(Layer):

    def __init__(self, input_dim, output_dim,
                 dropout=0., act=tf.nn.relu, bias=True,
                 batch_norm=False, bn_momentum=0.4, **kwargs):

        super(WordHashing, self).__init__(**kwargs)

        self._act = act
        self._bias = bias

        self._batch_norm = batch_norm


        with tf.variable_scope('WordHashing_vars'):

            self._tfvars['weights'] = glorot([input_dim, output_dim], get_var=True, name='weights', dtype=tf.float32)

            if self._bias:
                self._tfvars['bias'] = zeros([output_dim], get_var=True, name='bias')

    def _call(self, inputs, **kwargs):

        word_features = inputs

        dense_vector = tf.sparse.sparse_dense_matmul(word_features, self._tfvars['weights'])

        if self._bias:
            dense_vector += self._tfvars['bias']

        if self._batch_norm:
            dense_vector = tf.layers.batch_normalization(dense_vector, training=kwargs['bn_training'])

        output = self._act(dense_vector)

        return output

