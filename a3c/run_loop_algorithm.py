"""Execution of loop algorithm.
  this program maximally mimic play_icegame.
"""
from __future__ import print_function
import os
import sys
import argparse
import numpy as np

from envs import create_icegame_env

def LongLoopAlgorithm(args):
  outdir = os.path.join(args.out_dir, 'loopalgo')
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  # define environment
  env = create_icegame_env(outdir, args.env_id)
  env.start(create_defect=False)

  length = 0
  rewards = 0

  for ep in range(args.num_episodes):
    last_state = env.reset(create_defect=True)
    steps_rewards=[]
    
    while True:
      state, reward, terminate, info = env.auto_step()

      rewards += reward
      length += 1

      if terminate:
        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))
        length = 0
        rewards = 0
        break


def main():
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--out-dir', default=None, help='output log directory. Default: log_dir/inference/')
  parser.add_argument('--env-id', default="IcegameEnv-v0", help='Environment id')
  parser.add_argument('-p', '--policy', type=str, default='longloop', help='Choose algorithm: Only long loop or SSF.')
  parser.add_argument("-n", "--num-episodes", default=1000, type=int, help="Number of episodes to run.")

  args = parser.parse_args()
  LongLoopAlgorithm(args)

if __name__ == '__main__':
  main()