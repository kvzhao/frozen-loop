
import os
import sys
import gym
import gym_icegame
import numpy as np

#TODO: dangerous relative path
sys.path.append("../icegame2/build/src/")

def create_icegame_env(path, ID, hparams):
    """ Set environment parameters here.
      * Need args here. which implies shell commands enhancement.
    """
    if ID in ["IcegameEnv-v0", "IcegameEnv-v3", "FModelEnv-v3", 'f', 'sqice']:
        if ID == 'f':
            ID = "FModelEnv-v3"
        elif ID == 'sqice':
            ID = "IcegameEnv-v0"
        print ("Create Env {}".format(ID))
    else:
        print ("Env {} is not suitable for this project.".format(ID))
    env = gym.make(ID)
    if not os.path.exists(path):
        os.mkdir(path)
    env.set_output_path(path)

    # set the environments
    # Note: Directly include params? or passing into from main?
    env.set_training_condition(
        defect_upper_thres=hparams.defect_upper_thres,
        defect_lower_thres=hparams.defect_lower_thres,
        dconfig_amp=hparams.dconfig_amp,
        #local_eng_level=hparams.disable_local,
        smallsize_discount=hparams.smallsize_discount,
        failure_reward=hparams.failure_reward,
        accept_reward=hparams.accept_reward,
        stepwise_invfactor=hparams.stepwise_invfactor,
        config_refresh_steps=hparams.config_refresh_steps,
    )

    # save the settings
    env.dump_env_setting()
    # back up the initial configuration
    env.save_ice()
    return env

create_icegame_env("envLogs", "IcegameEnv-v0")
#create_icegame_env("envLogs", "FModelEnv-v3")
