import logging
from gym.envs.registration import register
logger = logging.getLogger(__name__)

# Move to the new icegame2
register (
    id = "IcegameEnv-v3",
    entry_point="gym_icegame.envs:IcegameEnv",
    # There would be more arguments
    kwargs={
        "L" : 16,
        "kT" : 0.001,
        "J" : 1.0,
        "defect_upper_thres" : 4,
        "defect_lower_thres" : 20,
        "dconfig_amp" : 5,
        "local_eng_level" : True,
        "stepwise_invfactor" : 1000.0,
        "config_refresh_steps" : 100000,
    },
)

# Move to the new icegame2
register (
    id = "FModelEnv-v3",
    entry_point="gym_icegame.envs:FModelGameEnv",
    # There would be more arguments
    kwargs={
        "L" : 16,
        "kT" : 0.001,
        "J1" : 1.0,
        "J2" : -0.05,
        "defect_upper_thres" : 4,
        "defect_lower_thres" : 20,
        "dconfig_amp" : 5,
        "local_eng_level" : True,
        "stepwise_invfactor" : 1000.0,
        "config_refresh_steps" : 100000,
    },
)