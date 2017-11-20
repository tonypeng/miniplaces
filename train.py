import argparse
import numpy as np
from Trainer import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net_name', type=str, default='resnet34')
parser.add_argument('--data_root', type=str, default='images')
parser.add_argument('--train_data_list', type=str, default='data/train.txt')
parser.add_argument('--train_data_h5', type=str, default='miniplaces_128_train.h5')
parser.add_argument('--val_data_list', type=str, default='data/val.txt')
parser.add_argument('--val_data_h5', type=str, default='miniplaces_128_val.h5')
parser.add_argument('--load_size', type=int, default=128)
parser.add_argument('--fine_size', type=int, default=128)
parser.add_argument('--center_mean', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default='momentum')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--rmsprop_decay', type=float, default=0.9)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epsilon', type=float, default=1e-2)
parser.add_argument('--iterations', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=150)
parser.add_argument('--dropout_keep_prob', type=float, default=0.8)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--val_loss_iter_print', type=int, default=10)
parser.add_argument('--checkpoint_iterations', type=int, default=1000)
args = parser.parse_args()

data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842]
                            if args.center_mean else np.zeros(3))

trainer = Trainer(args.net_name, args.data_root, args.train_data_list,
                    args.train_data_h5, args.val_data_list, args.val_data_h5,
                    args.load_size, args.fine_size,
                    data_mean, args.optimizer, args.learning_rate,
                    args.rmsprop_decay, args.momentum, args.epsilon,
                    args.iterations,
                    args.batch_size, args.dropout_keep_prob, args.device,
                    args.verbose, args.val_loss_iter_print,
                    args.checkpoint_iterations)
trainer.train()
