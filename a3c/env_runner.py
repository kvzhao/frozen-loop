from __future__ import print_function

from collections import namedtuple
import numpy as np
import tensorflow as tf
import six.moves.queue as queue
import distutils.version
import threading
import scipy
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

from configs import hparams
from params import HParams

"""ISSUES
    * LSTM policy and forward policy
    * state
    * timeout limit
"""

# Why we cannot use obs here?
Batch = namedtuple("Batch", ["gs", "ls", "a", "adv", "r", "terminal"])
#Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
        given a rollout, compute its returns and the advantage
    """
    # We should handle the list of dictionay
    #batch_si = np.asarray(rollout.states)

    batch_global_states = np.asarray(rollout.global_states)
    batch_local_states = np.asarray(rollout.local_states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    #features = rollout.features[0]
    return Batch(batch_global_states, batch_local_states, batch_a, batch_adv, batch_r, rollout.terminal)

class PartialRollout(object):
    """
        a piece of a complete rollout.  We run our agent, and process its experience
        once it has processed enough steps.
    """
    def __init__(self):
        #self.states = []
        self.global_states = []
        self.local_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, global_state, local_state, action, reward, value, terminal, features=None):
        #self.states += [state]
        self.local_states += [local_state]
        self.global_states+= [global_state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        """How to switch FFPolicy & LSTMPolicy easily?
        """
        #self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.global_states.extend(other.global_states)
        self.local_states.extend(other.local_states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        #self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
        One of the key distinctions between a normal environment and a universe environment
        is that a universe environment is _real time_.  This means that there should be a thread
        that would constantly interact with the environment and tell it what to do.  This thread is here.
    """

    def __init__(self, env, policy, hparams):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.hparams = hparams
        self.num_local_steps = hparams.local_steps

        self.env = env
        #self.last_features = None
        self.policy = policy

        self.daemon = True
        self.sess = None
        self.summary_writer = None

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        # start thread activity (threading)
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.
            self.queue.put(next(rollout_provider), timeout=600.0)

def env_runner(env, policy, num_local_steps, summary_writer):
    """
        The logic of the thread runner.  In brief, it constantly keeps on running
        the policy, and as long as the rollout exceeds a certain length, the thread
        runner appends the policy to the queue.
    """
    last_state = env.reset()
    #last_features = policy.get_initial_features()
    length = 0
    rewards = 0

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state)
            action, value_ = fetched[0], fetched[1]
            # for lstm policy
            #fetched = policy.act(last_state, *last_features)
            #action, value_, features = fetched[0], fetched[1], fetched[2:]
            # argmax to convert from one-hot

            #print ("action: {}".format(action))
            #print ("value _: {}".format(value_))

            # TODO: eps-greedy method
            state, reward, terminal, info = env.step(action.argmax())

            # collect the experience
            ls = last_state.local_obs
            gs = last_state.global_obs
            rollout.add(gs, ls, action, reward, value_, terminal)
            #rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward

            last_state = state
            #last_features = features

            # TODO: Enrich info in icegame
            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            """Avoid so ugly hard-coded timeout
                * Better timeout mechanism
            """
            timestep_limit = 1024
            if terminal or length >= timestep_limit:
                terminal_end = True
                # if length >= timestep_limit: # well, reset, anyway
                # It is important to reset the environment, Now just call reset handling for this.
                last_state = env.reset()
                #last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: {}, length: {}".format(rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state)
            #rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout