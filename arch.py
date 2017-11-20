import tensorflow as tf
import utils

def initialize_weights(shape, stddev):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def initialize_biases(shape, value):
    return tf.Variable(tf.constant(value, shape=shape))

def conv2d(x, ksize, stride, chan_out, border_mode='SAME', init_stddev=1.0):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    # no bias needed because we batch normalize
    W = initialize_weights([ksize, ksize, chan_in, chan_out], init_stddev)
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], border_mode)

def avg_pool(x, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)

def max_pool(x, ksize, stride=None, border_mode='SAME'):
    stride = stride or ksize
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding=border_mode)

def resid_block(x, is_training, ksize, activation=tf.nn.relu):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    conv1 = conv2d(x, ksize, 1, chan_in)
    conv1 = batch_norm(conv1, is_training)
    act1 = activation(conv1)

    conv2 = conv2d(act1, ksize, 1, chan_in)
    conv2 = batch_norm(conv2, is_training)

    return activation(conv2 + x)

def resid_block_dim_increase(x, is_training, ksize, stride, chan_out, activation=tf.nn.relu, init_stddev=1.0):
    chan_in = utils.tensor_shape_as_list(x)[-1]
    conv1 = conv2d(x, ksize, stride, chan_out, init_stddev=init_stddev)
    conv1 = batch_norm(conv1, is_training)
    act1 = activation(conv1)

    conv2 = conv2d(act1, ksize, 1, chan_out, init_stddev=init_stddev)
    conv2 = batch_norm(conv2, is_training)

    shortcut = conv2d(x, 1, stride, chan_out, init_stddev=init_stddev)
    return activation(conv2 + shortcut)

def resid_group(x, is_training, ksize, chan_out, num_resid, downsize_factor=2, init_stddev=1.0):
    resid1 = None
    chan_in = utils.tensor_shape_as_list(x)[-1]
    if chan_in == chan_out:
        resid1 = resid_block(x, is_training, ksize)
    else:
        resid1 = resid_block_dim_increase(x, is_training, ksize, downsize_factor, chan_out, init_stddev=init_stddev)

    outp = resid1
    for i in range(num_resid - 1):
        outp = resid_block(outp, is_training, ksize)
    return outp

def fully_connected(x, num_outputs, init_weights_stddev=1.0, init_bias_value=0.0):
    num_inputs = utils.tensor_shape_as_list(x)[1]
    W = initialize_weights([num_inputs, num_outputs], init_weights_stddev)
    b = initialize_biases([num_outputs], init_bias_value)
    return tf.nn.xw_plus_b(x, W, b)

def batch_norm(x, is_training, scale=False):
    return tf.contrib.layers.batch_norm(x, is_training=is_training, scale=scale)
