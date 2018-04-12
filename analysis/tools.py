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

import imageio
import numpy as np
"""Work Flow?
"""

def create_animation_from_traject():
  pass

class LogData(object):
  def __init__(self, data_path, output_dir, L=32):
    """Construct the data object from data list (time series)"""

    dlist = self.read_json(data_path)
    self.output_dir = output_dir
    
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
    print ("Load data from {} is done.".format(self.output_dir))

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

  def sample_loop_trajectory(self):
    traj = np.random.choice(self.loops)
    # maybe show some information
    print ("Sampled loop: \n length: {}".format(len(traj)))
    return traj
  
  def save_loopmap(self, loop, name):
    img = np.zeros(self.N)
    for l in loop:
      x, y = l % self.L, l // self.L
      if (x + y) % 2 == 0:
        img[l] = 1
      else:
        img[l] = -1
    plt.imshow(img.reshape(self.L, self.L), 'plasma', interpolation='None')

    fname = os.path.join(self.output_dir, name)
    plt.savefig(fname + ".png")
    print ("Save the loop image to {}.".format(fname))

  def save_heatmap(self, name):
    heatmap = np.zeros(self.N)
    total = 0
    for site, visted in self.visited_counter.items():
      total += visted
      heatmap[site] = visted
    heatmap /= total

    plt.imshow(heatmap.reshape(self.L, self.L))
    plt.title("Visting Heatmap 2D")
    plt.colorbar()

    fname = os.path.join(self.output_dir, name)
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

    fname = os.path.join(self.output_dir, name)
    plt.savefig(fname + "_3d.png")
    plt.clf()
    print ("Save the visting heatmap.")

    # save raw data out
    data_path = os.path.join(self.output_dir, "data")
    if not os.path.exists(data_path):
      os.mkdir(data_path)
    np.save(data_path + "/visited_counts", heatmap)
    print ("Save raw data to {}".format(data_path))

  def save_looplen_hist(self, name):
    lengths, counts = zip(*self.length_counter.items())
    total = np.sum(counts)
    weighted_sum = 0
    for length, count in self.length_counter.items():
      weighted_sum += length * count
    weighted_mean = weighted_sum / total
    width = 0.8
    plt.bar(lengths, counts, width, align='center', color='r')
    plt.title("Loop length Histogram")
    plt.xlabel("Loop Size")
    plt.ylabel("Counts")
    #plt.xticks(indices + width * 0.5, lengths)

    fname = os.path.join(self.output_dir, name)
    plt.savefig(fname + ".png")
    plt.clf()
    print ("Statistics: minlen : {}, Maxlen: {}, weighted mean: {}".format(
        np.min(lengths), np.max(lengths), weighted_mean))
    print ("Save the loop length histogram.")

    # save raw data out
    data_path = os.path.join(self.output_dir, "data")
    if not os.path.exists(data_path):
      os.mkdir(data_path)
    data_out = np.stack([lengths, counts], axis=1)
    np.save(data_path + "/length_hist_raw", data_out)
    print ("Save raw data to {}".format(data_path))

  def generate_loop_animation(self, loop, name):
    """Loop creation animation.
      * More information is needed, something like
        action, decision (Huge!), energy changes 
    """
    from matplotlib import animation
    images = []
    img = np.zeros(self.N)
    prev = 0
    fig = plt.figure()
    im = plt.imshow(img.reshape(self.L, self.L), 'plasma', interpolation='None')
    for idx in range(len(loop)):
      for lsite in loop[prev:idx]:
        # dummy loop, acutally is not a loop
        x, y = lsite % self.L, lsite//self.L
        img[lsite] = 1 if (x+y) %2 == 0 else -1
        im = plt.imshow(img.reshape(self.L, self.L), 'plasma', interpolation='None')
        images.append([im]) 
        # tricky: https://stackoverflow.com/questions/18019226/matplotlib-animation
      prev = idx
    print ("Done parsing the images, start making film...")

    ani = animation.ArtistAnimation(fig, images, 
          interval=20, blit=True, repeat_delay=500)

    fname = os.path.join(self.output_dir, name)
    ani.save(fname + ".mp4")
    print ("Save the loop animation to {}".format(fname))

