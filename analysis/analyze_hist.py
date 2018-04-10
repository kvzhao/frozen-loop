import argparse
import os

from tools import LogData

def main():
  parser = argparse.ArgumentParser(description='Analyise history from Icegame environment.')
  parser.add_argument('-l', '--logdir', type=str, default='env_history.json', help='Json log file path.')
  parser.add_argument('-o', '--output', type=str, default='results', help='Path to output folder')
  args = parser.parse_args()

  if not os.path.exists(args.output):
    os.mkdir(args.output)

  data = LogData(args.logdir)

  fname = '/'.join([args.output, 'length_hist'])
  data.save_looplen_hist(fname)

  fname = '/'.join([args.output, 'heatmap'])
  data.save_heatmap(fname)

if __name__ == '__main__':
  main()