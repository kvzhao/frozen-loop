import argparse
import os

from tools import LogData
from params import args

def main():

  if not os.path.exists(args.output):
    os.mkdir(args.output)

  data = LogData(args.logdir, args.output)

  fname = '/'.join([args.output, 'length_hist'])
  data.save_looplen_hist("length_hist")

  fname = '/'.join([args.output, 'heatmap'])
  data.save_heatmap("heatmap")

if __name__ == '__main__':
  main()