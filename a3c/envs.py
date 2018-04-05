
import os
import sys
import gym
import gym_icegame
import numpy as np

#TODO: dangerous relative path
sys.path.append("../icegame2/build/src/")

def create_icegame_env(path, ID):
    if ID in ["IcegameEnv-v0"]:
        print ("Create Env {}".format(ID))
    else:
        print ("Env {} is not suitable for this project.".format(ID))
    env = gym.make(ID)
    env.set_output_path(path)
    return env

create_icegame_env(".", "IcegameEnv-v0")