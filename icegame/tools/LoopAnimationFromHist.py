import matplotlib.pyplot as plt
import sys, os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Analyise history from Icegame environment.')

parser.add_argument('--logdir', type=str, default='env_history.json', help='Json log file path.')

args = parser.parse_args()

def read_json(filename):
    with open(filename, 'r') as json_file:
        d_list = [json.loads(line) for line in json_file]
        print ('Log contains {} timestamps.'.format(len(d_list)))
    return d_list

def site2coord(sites, L):
    coords = None
    if type(sites) == int:
        coords = (int(sites/L), int(sites%L))
        # return tuple
    elif type(sites) == list:
        coords = [] # return list
        for s in sites:
            coords.append((int(s/L), int(s%L)))
    return coords 

def plot_loops(loops2d):
    x, y = zip(*loops2d)
    plt.scatter(x, y, vmax=32)
    plt.title('Loop configurations')
    plt.savefig('loop_config.png')
    plt.show()

def plotloop(loop):
    map_ = np.zeros(1024)
    for idx, l in enumerate(loop):
        x, y = int(l/32), int(l%32)
        if (x+y) % 2 == 0:
            map_[l] = 1.0
        else:
            map_[l] = -1.0
    map2d = map_.reshape(32,32)
    plt.imshow(map2d, 'Reds', interpolation="None",  vmin=-1, vmax=1)
    plt.show()

def save_loop_animation(loop, name, path):
    map_ = np.zeros(1024)
    for idx, l in enumerate(loop):
        x, y = int(l/32), int(l%32)
        if (x+y) % 2 == 0:
            map_[l] = 1.0
        else:
            map_[l] = -1.0
        map2d = map_.reshape(32,32)
        plt.imshow(map2d, 'Reds', interpolation="None",  vmin=-1, vmax=1)
        plt.savefig("{}/{}_{}.png".format(path, name, idx))
    complete = np.zeros((32,32))
    plt.imshow(complete, 'Reds', interpolation="None",  vmin=-1, vmax=1)
    plt.savefig("{}/{}_{}.png".format(path, name, idx+1))
    plt.clf()

# then we should carefully parse list

data = read_json(args.logdir)
# extract data information
'''
    Several items stored in data
    * Episode
    * Steps
    * Trajectory (list)
    * EnclosedArea
    * ActionStats (list)
    * LoopLength
    * UpdateTimes
    * AcceptanceRatio
    * StartSite
'''

episodes = []
steps = []
updates = []
lengths = []
areas = []
accept_ratio = []
action_stats = []
loops = []
loops2d = []

indices = list(range(len(data)))

## extract data
for d in data:
    episodes.append(d['Episode'])
    steps.append(d['Steps'])
    lengths.append(d['LoopLength'])
    areas.append(d['EnclosedArea'])
    accept_ratio.append(d['AcceptanceRatio'])
    action_stats.append(d['ActionStats'])
    loops.append(d['Trajectory'])
    loops2d.extend(site2coord(d['Trajectory'], 32))

print ("Convert dict into list")
#areas, loops = zip(*sorted(zip(areas, loops), reverse=True))
#print (loops[-1])

counter = 0
for area, loop in zip(areas, loops):
    size = len(loop)
    if area >= 3 or size >= 30:
        save_loop_animation(loop, str(counter), "anime3")
        counter += 1