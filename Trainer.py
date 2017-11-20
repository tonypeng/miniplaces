import nets
import inception_v4

import tensorflow as tf
from DataLoader import *

class Trainer:
    def __init__(self, net_name, data_root, train_data_list, val_data_list,
                 load_size, fine_size, data_mean, optimizer, learning_rate,
                 rmsprop_decay, rmsprop_momentum, epsilon,
                 iterations,
                 batch_size, dropout_keep_prob, device, verbose=True,
                 val_loss_iter_print=10):
        self.net_name = net_name
        self.data_root = data_root
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.rmsprop_decay = rmsprop_decay
        self.rmsprop_momentum = rmsprop_momentum
        self.epsilon = epsilon
        self.iterations = iterations
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.load_size = load_size
        self.fine_size = fine_size
        self.data_mean = data_mean
        self.device = device
        self.verbose = verbose
        self.val_loss_iter_print = val_loss_iter_print

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
            is_training = tf.placeholder(tf.bool, name='is_training')

            net = self._construct_net(x, keep_dropout, is_training)

            losses = 0
            for outp in net:
                logits, weight = outp
                losses += weight * tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y, logits=logits)
            loss = tf.reduce_mean(losses)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer()
                optimize = optimizer.minimize(loss)

            accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(net[0][0], y, 1), tf.float32)) * 100
            accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(net[0][0], y, 5), tf.float32)) * 100

            sess.run(tf.global_variables_initializer())

            it = 0
            for it in range(self.iterations):
                images_batch, labels_batch = loader_train.next_batch(self.batch_size)

                sess.run(optimize,
                    feed_dict={
                        x: images_batch,
                        y: labels_batch,
                        keep_dropout: self.dropout_keep_prob,
                        is_training: True})
                curr_loss, acc1, acc5 = sess.run([loss, accuracy1, accuracy5],
                                                feed_dict={
                                                    x: images_batch,
                                                    y: labels_batch,
                                                    keep_dropout: self.dropout_keep_prob,
                                                    is_training: False})
                if it % self.val_loss_iter_print == 0:
                    images_batch_val, labels_batch_val = loader_val.next_batch(self.batch_size)
                    curr_val_loss, val_acc1, val_acc5 = sess.run([loss, accuracy1, accuracy5],
                                            feed_dict={
                                                x: images_batch_val,
                                                y: labels_batch_val,
                                                keep_dropout: self.dropout_keep_prob,
                                                is_training: False})
                    print("Iteration " + str(it + 1) + ": Training Loss=" + str(curr_loss) + "; Val Loss=" + str(curr_val_loss))
                    print("              Val Acc1=" + str(val_acc1) + "%; Val Acc5="+str(val_acc5)+"%")
                else:
                    print("Iteration " + str(it + 1) + ": Loss=" + str(curr_loss) + "; Acc1="+str(acc1)+"%; Acc5="+str(acc5)+"%")

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _construct_optimizer(self):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(
                        learning_rate=self.learning_rate,
                        decay=self.rmsprop_decay,
                        momentum=self.rmsprop_momentum,
                        epsilon=self.epsilon)
        raise NotImplementedError

    def _construct_net(self, inp, dropout_keep_prob, is_training):
        if self.net_name == 'alexnet':
            return [(nets.AlexNet(inp, dropout_keep_prob), 1.0)]
        elif self.net_name == 'inception':
            logits, end_points = inception_v4.inception_v4(inp, num_classes=100, dropout_keep_prob=self.dropout_keep_prob)
            return [(logits, 1.0), (end_points['AuxLogits'], 0.4)]
        elif self.net_name == 'resnet34':
            return [(nets.ResNet34(inp, is_training, {}), 1.0)]
        raise NotImplementedError
