import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def weight_variable(name, shape=None, dtype=tf.float32, initial_value=None):
    if initial_value is None:
        return tf.get_variable(name=name,
                               shape=shape,
                               dtype=dtype,
                               initializer=xavier_initializer())
    return tf.Variable(initial_value=initial_value, name=name, dtype=dtype)


def bias_variable(name, shape=None, dtype=tf.float32, initial_value=None):
    if initial_value is None:
        return tf.get_variable(name=name,
                               shape=shape,
                               dtype=dtype,
                               initializer=tf.constant_initializer(0.0001))
    return tf.Variable(initial_value=initial_value, name=name, dtype=dtype)


def mlp(inputs, dims, activation_fns, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h = inputs
        for layer, dim in enumerate(dims):
            h = tf.layers.dense(h, dim, activation_fns[layer], name="layer_" + str(layer))
    return h



