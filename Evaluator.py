import nets
import tensorflow as tf
from DataLoader import *

class Evaluator:
    def __init__(self, arch, model_names, model_paths, model_weights, data_root,
                    data_list, load_size, fine_size, batch_size, device,
                    data_mean, hidden_activation):
        self.arch = arch
        self.model_names = model_names
        self.model_paths = model_paths
        self.model_weights = model_weights
        self.data_root = data_root
        self.data_list = data_list
        self.load_size = load_size
        self.fine_size = fine_size
        self.batch_size = batch_size
        self.device = device
        self.data_mean = data_mean
        self.hidden_activation = hidden_activation

    def evaluate(self):
        # Construct dataloader
        opt_data_eval = {
            'data_root': self.data_root,
            'data_list': self.data_list,
            'data_mean': self.data_mean,
            'load_size': self.load_size,
            'fine_size': self.fine_size,
            'randomize': False
        }

        loader = DataLoaderDisk(**opt_data_eval)

        total_output = np.zeros(100)

        model_names = self.model_names.split(',')
        model_paths = self.model_paths.split(',')
        archs = self.arch.split(',')
        model_weights = map(int, self.model_weights.split(','))
        assert len(model_names) == len(model_paths) == len(archs)

        g = tf.Graph()
        with g.as_default(), g.device(self.device), tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            x = tf.placeholder(
                tf.float32, [None, self.fine_size, self.fine_size, 3])
            y = tf.placeholder(tf.int64, None)
            is_training = tf.placeholder(tf.bool, name='is_training')

            nets = []
            for model_name, model_weight, arch in zip(model_names,
                                                        model_weights,
                                                        archs):
                with tf.variable_scope(model_name):
                    nets.append(self._construct_net(
                                        arch, x, is_training) * model_weight)
            combined = tf.add_n(nets)

            sess.run(tf.global_variables_initializer())

            top5 = tf.nn.top_k(combined, k=5, sorted=True)
            acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(combined, y, 5), tf.float32)) * 100

            # Restore model
            for model_name, model_path in zip(model_names, model_paths):
                saver = tf.train.Saver(max_to_keep=5, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name))
                saver.restore(sess, model_path)

            img_num = 1
            while img_num <= loader.size():
                images_batch, labels_batch = loader.up_to(self.batch_size)
                # Evaluate
                top5_output, acc5_output = sess.run([top5, acc5], feed_dict={
                                        x: images_batch,
                                        y: labels_batch,
                                        is_training: False})
                for particular_top_5 in top5_output.indices:
                    print('test/'+str(img_num).zfill(8)+'.jpg '+(' '.join(str(c) for c in particular_top_5)))
                    img_num += 1


    def _construct_net(self, arch, inp, is_training):
        if arch == 'alexnet':
            return nets.AlexNet(inp, 1.0)
        elif arch == 'inception':
            logits, end_points = inception_v4.inception_v4(inp, num_classes=100, dropout_keep_prob=1.0)
            return logits
        elif arch == 'resnet18':
            return nets.ResNet18(inp, is_training, {
                        'hidden_activation': self.hidden_activation
                        })
        elif arch == 'resnet34':
            return nets.ResNet34(inp, is_training, {
                        'hidden_activation': self.hidden_activation
                        })
        elif arch == 'resnet50':
            return nets.ResNet50(inp, is_training, {
                        'hidden_activation': self.hidden_activation
                        })
        raise NotImplementedError
