import nets
import tensorflow as tf
import math
from DataLoader import *

class Ensemble:
    def __init__(self, model_name, model_path, data_root, data_list,
                    image_size, batch_size, device, data_mean,
                    hidden_activation):
        self.model_name = model_name
        self.model_path = model_path
        self.data_root = data_root
        self.data_list = data_list
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        self.data_mean = data_mean
        self.hidden_activation = hidden_activation
        self.test_data_h5 = 'miniplaces_128_test.h5'
        self.checkpoint_names = ['fiona', 'ryan', 'bili2', 'negan']
        self.archs = ['resnet18', 'resnet18', 'resnet34', 'resnet34']
        self.checkpoint_root = 'checkpoints/'
        self.arch = 'resnet18'


                # for particular_top_5 in top5_res:
                #     print('test/'+str(img_num).zfill(8)+'.jpg '+(' '.join(str(c) for c in particular_top_5.indices[0])))
                #     img_num += 1

    def evaluate(self):
        check_names = []
        for checkpoint_path, nm, arch in zip([self.checkpoint_root+n+'/' for n in self.checkpoint_names], self.checkpoint_names, self.archs):
            if tf.gfile.IsDirectory(checkpoint_path):
                path = tf.train.latest_checkpoint(checkpoint_path)
            check_names.append((path, nm, arch))

        output_list = []
        # opt
        opt_data_eval = {
            'data_h5': self.test_data_h5,
            'data_root': self.data_root,
            'data_list': self.data_list,
            'data_mean': self.data_mean,
            'load_size': self.image_size,
            'fine_size': self.image_size,
            'randomize': False
        }

        for cpath, mname, arch in check_names:
            g = tf.Graph()
            with g.as_default(), tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                x = tf.placeholder(
                    tf.float32, [None, self.image_size, self.image_size, 3])
                is_training = tf.placeholder(tf.bool, name='is_training')

                self.arch = arch
                with tf.variable_scope(mname):
                    net = self._construct_net(x, is_training)

                sess.run(tf.global_variables_initializer())

                num_batches = int(math.ceil(10000 / float(self.batch_size)))
                total_output = np.empty([num_batches * self.batch_size, 100])
                offset = 0
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    logits = tf.cast(tf.constant(output_list[i]), dtype=tf.float32)
                    saver = tf.train.Saver(max_to_keep=5)
                    saver.restore(sess, cpath)
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    loader = DataLoaderH5(**opt_data_eval)
                    for i in range(num_batches):
                        images_batch, labels_batch = loader.next_batch(self.batch_size)
                        print('step: %d/%d' % (i, num_batches))
                        o = sess.run([logits], feed_dict={
                                                x: images_batch,
                                                is_training: False})
                        total_output[offset:offset + self.batch_size] = o
                        offset += self.batch_size
                    coord.request_stop()
                    coord.join(threads)
                output_list.append(total_output)
                labels_list.append(total_labels)

        total_count = num_batches * self.batch_size

        for i in range(len(output_list)):
            predictions = tf.nn.softmax(logits)
            labels = tf.constant(labels_list[i])
            top1_op = tf.nn.in_top_k(predictions, labels, 1)
            top5_op = tf.nn.in_top_k(predictions, labels, 5)

            with tf.Session() as sess:
                top1, top5 = sess.run([top1_op, top5_op])

            print('Top 1 accuracy: %f' % (np.sum(top1) / float(total_count)))
            print('Top 5 accuracy: %f' % (np.sum(top5) / float(total_count)))

        output_sum = tf.zeros([total_count, dataset.num_classes])
        for output in output_list:
            logits = tf.cast(tf.constant(output), dtype=tf.float32)
            output_sum += logits
        output_sum /= len(output_list)

        predictions = tf.nn.softmax(output_sum)
        labels = tf.constant(labels_list[0])
        top1_op = tf.nn.in_top_k(predictions, labels, 1)
        top5_op = tf.nn.in_top_k(predictions, labels, 5)

        with tf.Session() as sess:
            top1, top5 = sess.run([top1_op, top5_op])

        print('Top 1 accuracy: %f' % (np.sum(top1) / float(total_count)))
        print('Top 5 accuracy: %f' % (np.sum(top5) / float(total_count)))

    def _construct_net(self, inp, is_training):
        if self.arch == 'alexnet':
            return nets.AlexNet(inp, 1.0)
        elif self.arch == 'inception':
            logits, end_points = inception_v4.inception_v4(inp, num_classes=100, dropout_keep_prob=1.0)
            return logits
        elif self.arch == 'resnet18':
            return nets.ResNet18(inp, is_training, {
                        'hidden_activation': self.hidden_activation
                        })
        elif self.arch == 'resnet34':
            return nets.ResNet34(inp, is_training, {
                        'hidden_activation': self.hidden_activation
                        })
        raise NotImplementedError
