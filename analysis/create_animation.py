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

  # we can save a gallery, sample several loops
  loop = data.sample_loop_trajectory()
  fname = os.path.join(args.output, "loop")
  data.save_loopmap(loop, fname)

  fname = os.path.join(args.output, "loop_animation")
  data.generate_loop_animation(loop, fname)

if __name__ == '__main__':
  main()