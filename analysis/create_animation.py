import argparse
import os

from tools import LogData

def main():
  parser = argparse.ArgumentParser(description='Analyise history from Icegame environment.')
  parser.add_argument('-l', '--logdir', type=str, default='env_history.json', help='Json log file path.')
  parser.add_argument('-o', '--output', type=str, default='results', help='Path to output folder.')
  parser.add_argument('-n', '--num_samples', type=int, default=1, help='Number of sampled trajectory.')
  args = parser.parse_args()

  if not os.path.exists(args.output):
    os.mkdir(args.output)

  data = LogData(args.logdir, args.output)

  # we can save a gallery, sample several loops
  
  for i in range(args.num_samples):
    loop = data.sample_loop_trajectory()
    data.save_loopmap(loop, "loop_{}".format(i))
    data.generate_loop_animation(loop, "loop_animation_{}".format(i))

if __name__ == '__main__':
  main()