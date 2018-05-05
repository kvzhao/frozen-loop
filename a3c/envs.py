
import os
import sys
import gym
import gym_icegame
import numpy as np

#TODO: dangerous relative path
sys.path.append("../icegame2/build/src/")

def create_icegame_env(path, ID):
    """ Set environment parameters here.
      * Need args here. which implies shell commands enhancement.
    """
    if ID in ["IcegameEnv-v3", "FModelEnv-v3"]:
        print ("Create Env {}".format(ID))
    else:
        print ("Env {} is not suitable for this project.".format(ID))
    env = gym.make(ID)
    if not os.path.exists(path):
        os.mkdir(path)
    env.set_output_path(path)

    # set the environments
    # Note: Directly include params? or passing into from main?

    # save the settings
    env.dump_env_setting()
    # back up the initial configuration
    env.save_ice()
    return env

create_icegame_env("envLogs", "IcegameEnv-v3")
