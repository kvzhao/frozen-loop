import argparse
import os

from tools import LogData
from params import args

def main():

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