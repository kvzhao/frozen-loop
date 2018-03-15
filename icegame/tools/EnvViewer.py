import matplotlib.pyplot as plt
import sys, os
import numpy as np

import gym
import gym_icegame
env = gym.make('IceGameEnv-v0')

def state_display(env):
    screens = []
    channels = env.observation_space.shape[-1]
    for i in range(channels):
        screens.append(env.get_obs()[:,:,i])
    num_screen = len(screens)
    fig = plt.figure()
    for n, screen in enumerate(screens):
        print (n+1)
        a = fig.add_subplot(1, num_screen, n + 1)
        plt.imshow(screen, 'Blues', interpolation=None)
    fig.set_size_inches(np.array(fig.get_size_inches()) * num_screen)
    plt.show()
