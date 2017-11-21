import nets
import tensorflow as tf
from DataLoader import *

class Evaluator:
    def __init__(self, arch, model_name, model_path, data_root, data_list,
                    image_size, batch_size, device, data_mean,
                    hidden_activation):
        self.arch = arch
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

    def evaluate(self):
        # Construct dataloader
        opt_data_eval = {
            'data_h5': self.test_data_h5,
            'data_root': self.data_root,
            'data_list': self.data_list,
            'data_mean': self.data_mean,
            'load_size': self.image_size,
            'fine_size': self.image_size,
            'randomize': False
        }

        loader = DataLoaderH5(**opt_data_eval)

        g = tf.Graph()
        with g.as_default(), g.device(self.device), tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, 3])
            is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.variable_scope(self.model_name):
                net = self._construct_net(x, is_training)

            sess.run(tf.global_variables_initializer())

            top5 = tf.nn.top_k(net, k=5, sorted=True)

            # Restore model
            saver = tf.train.Saver(max_to_keep=5)
            saver.restore(sess, self.model_path)

            img_num = 0
            while img_num < loader.size():
                images_batch, labels_batch = loader.next_batch(self.batch_size)

                # Evaluate
                top5_res = sess.run([top5], feed_dict={
                                        x: images_batch,
                                        is_training: False})

                for particular_top_5 in top5_res:
                    print('test/'+str(img_num).zfill(8)+'.jpg '+(' '.join(str(c) for c in particular_top_5.indices[0])))
                    img_num += 1

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
