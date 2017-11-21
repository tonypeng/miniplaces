import nets
import os
import inception_v4
import random
import tensorflow as tf
from DataLoader import *

class Trainer:
    def __init__(self, net_name, data_root, train_data_list, train_data_h5, val_data_list, val_data_h5,
                 load_size, fine_size, data_mean, optimizer, learning_rate, min_learning_rate, hidden_activation,
                 rmsprop_decay, momentum, epsilon, weight_decay,
                 iterations,
                 batch_size, dropout_keep_prob, device, verbose,
                 train_loss_iter_print, val_loss_iter_print, checkpoint_iterations,
                 checkpoint_name, start_from_iteration, model_name, log_path,
                 loss_adjustment_sample_interval, loss_adjustment_factor, loss_adjustment_coin_flip_prob):
        self.net_name = net_name
        self.data_root = data_root
        self.train_data_list = train_data_list
        self.train_data_h5 = train_data_h5
        self.val_data_list = val_data_list
        self.val_data_h5 = val_data_h5
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.hidden_activation = hidden_activation
        self.rmsprop_decay = rmsprop_decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.iterations = iterations
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.load_size = load_size
        self.fine_size = fine_size
        self.data_mean = data_mean
        self.device = device
        self.verbose = verbose
        self.train_loss_iter_print = train_loss_iter_print
        self.val_loss_iter_print = val_loss_iter_print
        self.checkpoint_iterations = checkpoint_iterations
        self.checkpoint_name = checkpoint_name
        self.start_from_iteration = start_from_iteration
        self.model_name = model_name
        self.log_path = log_path
        self.loss_adjustment_sample_interval = loss_adjustment_sample_interval
        self.loss_adjustment_factor = loss_adjustment_factor
        self.loss_adjustment_coin_flip_prob = loss_adjustment_coin_flip_prob
        self.loss_history = []

    def train(self):
        # =================================
        # This Is Where The Sausage Is Made
        # =================================

        # Construct dataloader
        opt_data_train = {
            'data_h5':self.train_data_h5,
            'data_root': self.data_root,
            'data_list': self.train_data_list,
            'load_size': self.load_size,
            'fine_size': self.fine_size,
            'data_mean': self.data_mean,
            'randomize': True
        }
        opt_data_val = {
            'data_h5': self.val_data_h5,
            'data_root': self.data_root,
            'data_list': self.val_data_list,
            'load_size': self.load_size,
            'fine_size': self.fine_size,
            'data_mean': self.data_mean,
            'randomize': False
        }
        # loader_train = DataLoaderDisk(**opt_data_train)
        # loader_val = DataLoaderDisk(**opt_data_val)
        loader_train = DataLoaderH5(**opt_data_train)
        loader_val = DataLoaderH5(**opt_data_val)

        curr_learning_rate = self.learning_rate

        path_save = './checkpoints/'+self.model_name+'/'

        g = tf.Graph()
        with g.as_default(), g.device(self.device), tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            learning_rate = tf.placeholder(tf.float32, None)
            x = tf.placeholder(
                tf.float32, [None, self.fine_size, self.fine_size, 3])
            y = tf.placeholder(tf.int64, None)
            keep_dropout = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool, name='is_training')

            x_preproc = tf.cond(is_training, lambda: self._preprocess_data(x), lambda: x)

            with tf.variable_scope(self.model_name):
                net = self._construct_net(x_preproc, keep_dropout, is_training)

            losses = 0
            for outp in net:
                logits, weight = outp
                losses += weight * tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y, logits=logits)
            loss = tf.reduce_mean(losses)
            if self.weight_decay is not None:
                regularizer = tf.add_n(tf.get_collection('weight_regularizers'))
                loss = loss + regularizer

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = self._construct_optimizer(learning_rate)
                optimize = optimizer.minimize(loss)


            sess.run(tf.global_variables_initializer())

            accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(net[0][0], y, 1), tf.float32)) * 100
            accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(net[0][0], y, 5), tf.float32)) * 100

            learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
            loss_training_summary = tf.summary.scalar('loss_training', loss)
            loss_valid_summary = tf.summary.scalar('loss_validation', loss)
            acc1_training_summary = tf.summary.scalar('training_accuracy1', accuracy1)
            acc5_training_summary = tf.summary.scalar('training_accuracy5', accuracy5)
            acc1_valid_summary = tf.summary.scalar('valid_accuracy1', accuracy1)
            acc5_valid_summary = tf.summary.scalar('valid_accuracy5', accuracy5)
            writer = tf.summary.FileWriter(os.path.join(self.log_path, self.model_name), graph=tf.get_default_graph())

            saver = tf.train.Saver(max_to_keep=5)

            it = 0
            if len(self.checkpoint_name)>1:
                saver.restore(sess, self.checkpoint_name)
                it = self.start_from_iteration

            while(it < self.iterations):
                images_batch, labels_batch = loader_train.next_batch(self.batch_size)

                sess.run(optimize,
                    feed_dict={
                        x: images_batch,
                        y: labels_batch,
                        learning_rate: curr_learning_rate,
                        keep_dropout: self.dropout_keep_prob,
                        is_training: True})

                if it % self.val_loss_iter_print == 0:
                    images_batch_val, labels_batch_val = loader_val.next_batch(self.batch_size)
                    curr_val_loss, val_acc1, val_acc5, val_loss_summ, val_acc1_summ, val_acc5_summ, learning_rate_summ = sess.run([loss, accuracy1, accuracy5, loss_valid_summary, acc1_valid_summary, acc5_valid_summary, learning_rate_summary],
                                            feed_dict={
                                                x: images_batch_val,
                                                y: labels_batch_val,
                                                learning_rate: curr_learning_rate,
                                                keep_dropout: self.dropout_keep_prob,
                                                is_training: False})

                    # adjust loss if we need to                                                                                                                                                                                        │··············
                    if self._should_adjust_learning_rate(curr_val_loss) and curr_learning_rate > 5e-5:
                        print ("Dropping learning rate from: " + str(curr_learning_rate))
                        curr_learning_rate = curr_learning_rate/self.loss_adjustment_factor
                        curr_learning_rate = max(curr_learning_rate, self.min_learning_rate)
                        print ("                       to: " + str(curr_learning_rate))
                    writer.add_summary(val_loss_summ, it)
                    writer.add_summary(val_acc1_summ, it)
                    writer.add_summary(val_acc5_summ, it)
                    writer.add_summary(learning_rate_summ, it)

                    print("Iteration " + str(it + 1) + ": Val Loss=" + str(curr_val_loss) + "%; Val Acc1=" + str(val_acc1) + "%; Val Acc5="+str(val_acc5)+"%")
                if it % self.train_loss_iter_print == 0:
                    curr_loss, acc1, acc5, loss_summ, acc1_summ, acc5_summ = sess.run([loss, accuracy1, accuracy5, loss_training_summary, acc1_training_summary, acc5_training_summary],
                                                    feed_dict={
                                                        x: images_batch,
                                                        y: labels_batch,
                                                        learning_rate: curr_learning_rate,
                                                        keep_dropout: self.dropout_keep_prob,
                                                        is_training: False})


                    writer.add_summary(loss_summ, it)
                    writer.add_summary(acc1_summ, it)
                    writer.add_summary(acc5_summ, it)

                    print("Iteration " + str(it + 1) + ": Loss=" + str(curr_loss) + "; Acc1="+str(acc1)+"%; Acc5="+str(acc5)+"%")
                if it % self.checkpoint_iterations == 0:
                    saver.save(sess, path_save, global_step=it)
                    print("Model saved at Iter %d !" %(it))
                it += 1


    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _construct_optimizer(self, learning_rate):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(
                        learning_rate=learning_rate,
                        decay=self.rmsprop_decay,
                        momentum=self.rmsprop_momentum,
                        epsilon=self.epsilon)
        elif self.optimizer == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate, self.momentum)
        raise NotImplementedError

    def _construct_net(self, inp, dropout_keep_prob, is_training):
        if self.net_name == 'alexnet':
            return [(nets.AlexNet(inp, dropout_keep_prob), 1.0)]
        elif self.net_name == 'inception':
            logits, end_points = inception_v4.inception_v4(inp, num_classes=100, dropout_keep_prob=self.dropout_keep_prob)
            return [(logits, 1.0), (end_points['AuxLogits'], 0.4)]
        elif self.net_name == 'resnet18':
            return [(nets.ResNet18(inp, is_training, {
                        'lambda': self.weight_decay,
                        'hidden_activation': self.hidden_activation
                        }), 1.0)]
        elif self.net_name == 'resnet34':
            return [(nets.ResNet34(inp, is_training, {
                        'lambda': self.weight_decay,
                        'hidden_activation': self.hidden_activation
                        }), 1.0)]
        elif self.net_name == 'resnet50':
            return [(nets.ResNet50(inp, is_training, {
                        'lambda': self.weight_decay,
                        'hidden_activation': self.hidden_activation
                        }), 1.0)]
        raise NotImplementedError

    def _preprocess_data(self, inp):
        p = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        pred = tf.less(p, 0.5)

        def _distort_image(inp):
            outp = tf.map_fn(lambda x: tf.image.random_hue(x, max_delta = .1), inp)
            outp = tf.map_fn(lambda x: tf.image.random_brightness(x, max_delta = .1), outp)
            outp = tf.map_fn(lambda x: tf.image.random_contrast(x, lower=.8, upper=1.2), outp)
            outp = tf.map_fn(lambda x: tf.image.random_saturation(x, lower=.8, upper=1.2), outp)
            return outp

        return tf.cond(pred, lambda: _distort_image(inp), lambda: inp)

    def _should_adjust_learning_rate(self, val_loss):
        self.loss_history.append(val_loss)
        if len(self.loss_history) > 3*self.loss_adjustment_sample_interval:
            self.loss_history.pop(0)
            old_loss = sum(self.loss_history[:self.loss_adjustment_sample_interval])/self.loss_adjustment_sample_interval
            recent_loss = sum(self.loss_history[2*self.loss_adjustment_sample_interval:])/self.loss_adjustment_sample_interval
            if recent_loss > old_loss:
                self.loss_history = []
                return random.uniform(0, 1) < self.loss_adjustment_coin_flip_prob
        return False
