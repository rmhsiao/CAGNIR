
import tensorflow as tf

from .inits import zeros, glorot

from collections import defaultdict


# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/williamleif/GraphSAGE
# which is under an identical MIT license as CAGNIR



_LAYER_UIDS = defaultdict(lambda:0)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):

        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = '%s_%s' % (layer, _LAYER_UIDS[layer])
        self.name = name
        self._tfvars = {}  # 用來設置tf變數，可方便_log_vars使用
        self._log_tfvars = kwargs.get('log_tfvars', False)
        # self._sparse_inputs = False

    def _call(self, inputs, **kwargs):
        return inputs

    def __call__(self, inputs, **kwargs):

        outputs = self._call(inputs, **kwargs)

        if self._log_tfvars:

            with tf.variable_scope('vars'):

                for var_name, var in self._tfvars.items():
                    tf.summary.histogram('%s/%s'%(self.name, var_name), tf.where(tf.is_nan(var), tf.zeros_like(var), var))

            tf.summary.histogram('%s/outputs'%self.name, outputs)

        return outputs


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., 
                 act=tf.nn.relu, bias=True, #featureless=False, sparse_inputs=False
                 batch_norm=True, # bn_momentum=0.4,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)

        self._act = act
        self._bias = bias

        self._batch_norm = batch_norm  # for batch normalization


        with tf.variable_scope(self.name+'_vars'):

            self._tfvars['weights'] = glorot((input_dim, output_dim), get_var=True, name='weights')

            if self._bias:
                self._tfvars['bias'] = zeros([output_dim], get_var=True, name='bias')


    def _call(self, inputs, **kwargs):

        x = inputs

        trans = tf.matmul(x, self._tfvars['weights'])

        if self._bias:
            trans += self._tfvars['bias']

        if self._batch_norm:
            trans = tf.layers.batch_normalization(trans, training=kwargs['bn_training'])

        return self._act(trans)


def build_mlp(input_layer, hid_units, batch_norm, bn_training, log_tfvars, name, last_act=False):

    prev_layer_outputs = input_layer

    for dense_seq, output_dim in enumerate(hid_units):

        input_dim = int(prev_layer_outputs.shape[1])
        act = tf.nn.relu if ((dense_seq+1)!=len(hid_units) or last_act) else lambda x: x

        layer_name = '%s.%s'%(name, dense_seq)

        with tf.variable_scope(layer_name):

            dense_layer = Dense(input_dim, output_dim, act=act, batch_norm=batch_norm, name=layer_name, log_tfvars=log_tfvars)
            prev_layer_outputs = dense_layer(prev_layer_outputs, bn_training=bn_training)

    return tf.identity(prev_layer_outputs, name=name)
