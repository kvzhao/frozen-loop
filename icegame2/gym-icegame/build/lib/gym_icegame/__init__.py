import logging
from gym.envs.registration import register
logger = logging.getLogger(__name__)

# Move to the new icegame2
register (
    id = "IcegameEnv-v0",
    entry_point="gym_icegame.envs:IcegameEnv",
    # There would be more arguments
    kwargs={
        "L" : 16,
        "kT" : 0.0001,
        "J" : 1.0,
        "stepwise_reward": "constant",
        "end_reward": "loopsize",
        "terminate_mode": "metro",
        "obs_type" : "multi",
    },
    )
