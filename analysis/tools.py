import os
import json
import argparse

import collections
from collections import Counter

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
"""Work Flow?

  How to do animation for !
"""

def create_animation_from_traject():
  pass

class LogData(object):
  def __init__(self, hist_filename, L=32):
    """Construct the data object from data list (time series)"""

    dlist = self.read_json(hist_filename)
    
    self.steps=[]
    self.episodes=[]
    self.lengths=[]
    self.loops=[] # list of lists
    self.visited_sites=[]
    self.L = L
    self.N = L ** 2

    for d in dlist:
      self.steps.append(d["Steps"])
      self.lengths.append(d["LoopLength"])
      self.loops.append(d["Trajectory"])
      self.visited_sites.extend(d["Trajectory"])

    self.steps = np.asarray(self.steps)
    self.lengths = np.asarray(self.lengths)
    self.length_counter = Counter(self.lengths)
    self.visited_counter = Counter(self.visited_sites)

    self._show_info()
    print ("Load data from {} is done.".format(hist_filename))

  def read_json(self, filename):
    """read from json returns list of lines
    """
    with open(filename, 'r') as json_file:
      d_list = [json.loads(line) for line in json_file]
      print ('Log contains {} timestamps.'.format(len(d_list)))
    return d_list


  def _show_info(self):
    # Call when init is done.
    pass

  def save_heatmap(self, fname):
    heatmap = np.zeros(self.N)
    total = 0
    for site, visted in self.visited_counter.items():
      total += visted
      heatmap[site] = visted
    heatmap /= total

    plt.imshow(heatmap.reshape(self.L, self.L))
    plt.title("Visting Heatmap 2D")
    plt.colorbar()
    plt.savefig(fname + "_2d.png")
    plt.clf()

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(111, projection='3d')
    _x = np.arange(self.L)
    _y = np.arange(self.L)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = heatmap
    bottom = np.zeros_like(top)
    width = depth = 1
    ax1.bar3d(x, y, bottom, width, depth, top, color='r')
    plt.title("Visting Heatmap 3D")
    plt.savefig(fname + "_3d.png")
    plt.clf()
    print ("Save the visting heatmap.")

  def save_looplen_hist(self, fname):
    lengths, counts = zip(*self.length_counter.items())
    indices = np.arange(len(counts))
    width = 0.8
    plt.bar(lengths, counts, width, align='center', color='r')
    plt.title("Loop length Histogram")
    plt.xlabel("Loop Size")
    plt.ylabel("Counts")
    #plt.xticks(indices + width * 0.5, lengths)
    plt.savefig(fname + ".png")
    plt.clf()
    """
    df = pd.DataFrame.from_dict(self.length_counter, orient='index')
    bar = df.plot(kind='bar')
    fig = bar.get_figure()
    fig.savefig(fname + ".png")
    """

    print ("Save the loop length histogram.")