import argparse
import Trainer

parser = argparse.ArgumentParser(description='')
parser.add_argument('--learning_rate', type=float, default=0.1)
args = parser.parse_args()

trainer = Trainer(args.learning_rate)
trainer.train()