import sys
import os
sys.path.append("build/src")

## ice game
import gym
import gym_icegame
import numpy as np

env = gym.make("IceGameEnv")

logdir = "logs_randombot"
num_of_episodes = 10

if not os.path.exists(logdir):
    os.makedirs(logdir)
env.set_output_path(logdir)

def random_policy(state):
    # Just randomly choose an action 
    # w/o considering observed states.
    action = np.random.randint(7)
    return action

print ("Running {} episodes...".format(num_of_episodes))

# Put the agent on the lattice
obs = env.start()
reward = 0.0
for ep in range(num_of_episodes):

    terminate = False
    # cumulant rewards
    ep_reward = 0.0

    while(not terminate):
        act = random_policy(obs)
        obs, reward, terminate, rets = env.step(act)
        ep_reward += reward

        obsvec = obs["configs_1d"]
        print (obsvec)

        if (terminate):
            print ("Should Reset")

    print ("Episode {}: Reward = {}".format(ep, ep_reward))

    _ = env.reset()

print ("Dump the reults to log file")
env.dump_env_status()

print ("Please run:\n python tools/AnalyiseHist.py --logdir=logs_randombot/env_history.json\n to see results.")
