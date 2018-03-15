import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register (
    id = "IceGameEnv",
    entry_point="gym_icegame.envs:IceGameEnv",
    # There would be more arguments
    kwargs={
        "L" : 32,
        "kT" : 0.0001,
        "J" : 1.0,
        "stepwise": "constant",
        "endreward": "loopsize",
        "terminate_mode": "metro",
        "obs_type" : "multi",
    },
    )