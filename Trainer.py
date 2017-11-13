import nets

import tensorflow as tf
from DataLoader import *

class Trainer:
    def __init__(self, net_name, data_root, train_data_list, val_data_list,
                 load_size, fine_size, data_mean, learning_rate, iterations,
                 batch_size, dropout_keep_prob, device, verbose=True):
        self.net_name = net_name
        self.data_root = data_root
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob,
        self.load_size = load_size
        self.fine_size = fine_size
        self.data_mean = data_mean
        self.device = device
        self.verbose = verbose

    def train(self):
        # =================================
        # This Is Where The Sausage Is Made
        # =================================

        # Construct dataloader
        opt_data_train = {
            'data_root': self.data_root,
            'data_list': self.train_data_list,
            'load_size': self.load_size,
            'fine_size': self.fine_size,
            'data_mean': self.data_mean,
            'randomize': True
        }
        opt_data_val = {
            'data_root': self.data_root,
            'data_list': self.val_data_list,
            'load_size': self.load_size,
            'fine_size': self.fine_size,
            'data_mean': self.data_mean,
            'randomize': False
        }

        loader_train = DataLoaderDisk(**opt_data_train)
        loader_val = DataLoaderDisk(**opt_data_val)

        g = tf.Graph()
        with g.as_default(), g.device(self.device), tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(
                tf.float32, [None, self.fine_size, self.fine_size, 3])
            y = tf.placeholder(tf.int64, None)
            keep_dropout = tf.placeholder(tf.float32)

            net = self._construct_net()

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y, logits=net))
            optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            it = 0
            for it in range(self.iterations):
                images_batch, labels_batch = loader_train.next_batch(self.batch_size)

                _, curr_loss = sess.run([optimize, loss],
                                        feed_dict={
                                            x: images_batch,
                                            y: labels_batch,
                                            keep_dropout: self.dropout_keep_prob})
                print("Iteration " + str(it + 1) + ": Loss=" + str(curr_loss))

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _construct_net(self, inp, dropout_keep_prob):
        if self.net_name == 'alexnet':
            return nets.AlexNet(inp, dropout_keep_prob)
        raise NotImplementedError