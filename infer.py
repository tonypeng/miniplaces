import argparse
import numpy as np
from Ensemble import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--model_path', type=str, default='model')
parser.add_argument('--data_root', type=str, default='data/images')
parser.add_argument('--data_list', type=str, default='data/test.txt')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='/gpu:0')
parser.add_argument('--center_mean', type=bool, default=True)
parser.add_argument('--hidden_activation', type=str, default='elu')
args = parser.parse_args()

data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842]
                            if args.center_mean else np.zeros(3))

ensemble = Ensemble(args.model_name, args.model_path,
                        args.data_root, args.data_list, args.image_size,
                        args.batch_size, args.device, data_mean,
                        args.hidden_activation)
ensemble.evaluate()
