import argparse
import numpy as np
from Evaluator import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--arch', type=str, default='resnet34')
parser.add_argument('--model_names', type=str, default='')
parser.add_argument('--model_paths', type=str, default='')
parser.add_argument('--data_root', type=str, default='data/images')
parser.add_argument('--data_list', type=str, default='data/test.txt')
parser.add_argument('--load_size', type=int, default=128)
parser.add_argument('--fine_size', type=int, default=110)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--center_mean', type=bool, default=True)
parser.add_argument('--hidden_activation', type=str, default='elu')
args = parser.parse_args()

data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842]) if args.center_mean else np.zeros(3)

evaluator = Evaluator(args.arch, args.model_names, args.model_paths,
                        args.data_root, args.data_list,
                        args.load_size, args.fine_size,
                        args.batch_size, args.device, data_mean,
                        args.hidden_activation)
evaluator.evaluate()
