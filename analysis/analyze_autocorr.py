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

def integrated_autocorr_time(acorr):
  N = len(acorr)
  tau = 0
  for t, c in enumerate(acorr):
    tau += (1 - t/N) * c
  tau = 2 * tau + 1
  return tau

def main():

  data = LogData(args.logdir, args.output)
  s0 = data.get_ice_configs()

  model = IceModel()
  model.set_ice(s0)

  observs = []

  data.set_end_index(100)

  # you had better cal defect den as they do.
  for i, loop in enumerate(data.loops):
    model.apply_loop(loop)
    st = model.get_ice()
    ##dd = inner_prod(st, s0)
    dd = model.get_symm_vertex()
    observs.append(dd)

  obs = np.asarray(observs)
  acorr = data.cal_autocorr(obs, "auto_corr")

  #tau = integrated_autocorr_time(acorr)
  #print ("Integrated autocorrelation time: {}".format(tau))

  import acor
  tau, mean, sigma = acor.acor(obs)
  print ("Acor: tau = {}, mean = {}, sigma = {}".format(tau, mean, sigma))



if __name__ == '__main__':
  main()