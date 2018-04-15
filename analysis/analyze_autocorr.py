"""Now, for testing first
"""

import os
from tools import LogData
from tools import IceModel
from params import args

import numpy as np

def inner_prod(x, y):
  if len(x) != len(y):
    return 0
  return sum(i[0] * i[1] for i in zip(x, y))

def main():

  data = LogData(args.logdir, args.output)
  s0 = data.get_ice_configs()

  model = IceModel()
  model.set_ice(s0)

  dots = []

  # you had better cal defect den as they do.
  for i, loop in enumerate(data.loops):
    model.apply_loop(loop)
    st = model.get_ice()
    dot = inner_prod(st, s0)
    dots.append(dot)

  obs = np.asarray(dots)
  data.cal_autocorr(obs, "auto_corr")


if __name__ == '__main__':
  main()