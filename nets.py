import numpy as np
import tensorflow as tf

from arch import *

_CONV_WEIGHT_STD_DEV = 0.1

def ResNet34(x, is_training, opt):
    # 7x7 conv, 64 features, stride 2
    conv_1 = conv2d(x, 7, 2, 64, init_stddev=_CONV_WEIGHT_STD_DEV)
    conv_1 = batch_norm(conv_1, is_training)
    conv_1 = tf.nn.relu(conv_1)

    # 3x3 max pooling, stride 2
    pool_1 = max_pool(conv1, 3, 2)

    # Resid group 1, 64 features, scale / 2
    resid_group_1 = resid_group(pool1, 3, 64, 3, downsize_factor=1, init_stddev=_CONV_WEIGHT_STD_DEV)
    # Resid group 2, 128 features, scale / 2
    resid_group_2 = resid_group(resid_group_1, 3, 128, 4, init_stddev=_CONV_WEIGHT_STD_DEV)
    # Resid group 3, 256 features, scale / 2
    resid_group_3 = resid_group(resid_group_2, 3, 256, 6, init_stddev=_CONV_WEIGHT_STD_DEV)
    # Resid group 4, 512 features, scale / 2
    resid_group_4 = resid_group(resid_group_3, 3, 512, 3, init_stddev=_CONV_WEIGHT_STD_DEV)

    # Global average pooling
    avg_pool = tf.reduce_mean(resid_group_4, [1, 2])

    # Fully-connected layer with 100 classes
    fc = fully_connected(avg_pool, 100, init_weights_stddev=0.01)

    return fc # output

def AlexNet(x, keep_dropout):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2. / (11 * 11 * 3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2. / (5 * 5 * 96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2. / (3 * 3 * 256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2. / (3 * 3 * 384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),

        'wf6': tf.Variable(tf.random_normal([7 * 7 * 256, 4096], stddev=np.sqrt(2. / (7 * 7 * 256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2. / 4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2. / 4096)))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros(96)),
        'bc2': tf.Variable(tf.zeros(256)),
        'bc3': tf.Variable(tf.zeros(384)),
        'bc4': tf.Variable(tf.zeros(256)),
        'bc5': tf.Variable(tf.zeros(256)),

        'bf6': tf.Variable(tf.zeros(4096)),
        'bf7': tf.Variable(tf.zeros(4096)),
        'bo': tf.Variable(tf.zeros(100))
    }

    # Conv + ReLU + LRN + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.add(tf.matmul(fc6, weights['wf6']), biases['bf6'])
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)

    # FC + ReLU + Dropout
    fc7 = tf.add(tf.matmul(fc6, weights['wf7']), biases['bf7'])
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])

    return out
