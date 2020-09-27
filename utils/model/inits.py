
import tensorflow as tf
import numpy as np


# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/williamleif/GraphSAGE
# which is under an identical MIT license as CAGNIR



def uniform(shape, scale=0.05, dtype=tf.float32, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=dtype)
    return tf.Variable(initial, name=name)


def glorot(shape, get_var=False, name=None, dtype=tf.float32):
    """Glorot & Bengio (AISTATS 2010) init."""

    init_range = np.sqrt(6.0/(int(shape[0])+shape[1]))

    if get_var:

        if name is None:
            raise Exception('name is not defined.')

        initial = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name=name, shape=shape, initializer=initial, dtype=dtype)

    else:

        initial = tf.contrib.layers.xavier_initializer()
        return tf.Variable(initial, dtype=dtype, name=name)


def zeros(shape, get_var=False, name=None, dtype=tf.float32):
    """All zeros."""
    if get_var:

        if name is None:
            raise Exception('name is not defined.')

        return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer(), dtype=dtype)

    else:

        initial = tf.zeros(shape, dtype=dtype)
        return tf.Variable(initial, name=name)

def ones(shape, name=None, dtype=tf.float32):
    """All ones."""
    initial = tf.ones(shape, dtype=dtype)
    return tf.Variable(initial, name=name)
