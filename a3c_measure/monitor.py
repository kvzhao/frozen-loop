import sys
import os
import itertools
import collections
import argparse
import numpy as np
import tensorflow as tf
import time
import signal

import models
from models import LSTMPolicy, FConvLSTMPolicy, FConvPolicy
from worker import FastSaver
from worker import cluster_spec

from envs import create_icegame_env

#TODO: remove dependencies of hparams
from params import HParams

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PolicyMonitor(object):
    """
        Helps evaluating a policy by running an episode in an environment,
            saving a video, and plotting summaries to Tensorboard.

            Sync from the global policy network and run eval.

            * save new 'policy' model (mean loop length)
    Args:
        env: environment to run in
        summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries

        add policy selections.
    """
    def __init__(self, env, policy, task, hparams):

        #self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
        #self.video_dir = os.path.abspath(args.video_dir)

        self.env = env
        self.hparams = hparams
        self.policy = policy
        self.task = task
        self.summary_writer = None

        """
        indir = os.path.join(args.log_dir, 'train')
        with open(indir + "/checkpoint", "r") as f:
            first_line = f.readline().strip()
            print ("first_line is : {}".format(first_line))
        ckpt = first_line.split(' ')[-1].split('/')[-1][:-1]
        ckpt = ckpt.split('-')[-1]
        self.ckpt = indir + '/model.ckpt-' + ckpt
        """

        # define environment
        local_space = env.local_observation_space.n
        global_space = env.global_observation_space.shape
        action_space = env.action_space.n

        worker_device = "/job:worker/task:{}/cpu:0".format(self.task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                """TODO: Add option for policy selection
                """
                if self.policy == "simple":
                    self.network = models.SimplePolicy(global_space, local_space, action_space, self.hparams)
                elif self.policy == "cnn":
                    self.network = models.CNNPolicy(global_space, local_space, action_space, self.hparams)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                if self.policy == "simple":
                    self.pi = models.SimplePolicy(global_space, local_space, action_space, self.hparams)
                elif self.policy == "cnn":
                    self.pi = models.CNNPolicy(global_space, local_space, action_space, self.hparams)
                self.pi.global_step = self.global_step

        # copy weights from the parameter server to the local model
        self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.pi.var_list, self.network.var_list)])

    def set_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def eval(self, sess, num_episodes=200):
        """Same as process in worker
        """
        logger.info("Start eval policy")
        sess.run(self.sync)  # copy weights from shared to local

        # Run an episode
        #for _ in range(num_episodes):
        #    pass
        done = False

        last_state = self.env.reset()
        #last_features = self.policy.get_initial_features()

        total_reward = 0.0
        episode_length = 0
        policy_hist = np.zeros(self.env.action_space.n)

        # run #num of inferences
        # TODO: use params
        for _ in range(100):
            while not done:
                fetched = self.pi.act(last_state)
                action_probs, value_, = fetched[0], fetched[1]
                #fetched = self.policy.act(last_state, *last_features)
                #action_probs, value_, features = fetched[0], fetched[1], fetched[2:]

                # Greedy action when testing
                action = action_probs.argmax()
                state, reward, done, info = self.env.step(action)

                episode_length += 1
                total_reward += reward
                policy_hist[action] += 1

                last_state = state
                #last_features = features
            self.env.reset()
        total_reward /= 100
        episode_length /= 100
        self.env.sim.reset_config()

        # Add summaries
        policy_hist /= np.sum(policy_hist)
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
        episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
        for actidx, prob_a in enumerate(policy_hist):
            episode_summary.value.add(simple_value=prob_a, tag="eval/prob_a{}".format(actidx))
        self.summary_writer.add_summary(episode_summary, self.global_step.eval())
        self.summary_writer.flush()

        logger.info ("Eval results at step {}: total_reward {}, episode_length {}".format(self.global_step.eval(), total_reward, episode_length))

        return total_reward, episode_length

def run_monitor(args, server):
    logger.info("Execute run monitor")
    env = create_icegame_env(args.logdir, args.env_id, args)
    monitor = PolicyMonitor(env, args.policy, args.task, args)

    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    # print trainable variables
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    logdir = os.path.join(args.logdir, 'eval')
    summary_writer = tf.summary.FileWriter(logdir)

    monitor.set_writer(summary_writer)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                            logdir=logdir,
                            summary_op=None,
                            init_op=init_op,
                            init_fn=init_fn,
                            summary_writer=summary_writer,
                            ready_op=tf.report_uninitialized_variables(variables_to_save),
                            global_step=monitor.global_step)

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        logger.info("PE Session Entered")
        sess.run(monitor.sync)
        global_step = sess.run(monitor.global_step)
        logger.info("Starting monitoring at step=%d", global_step)

        while not sv.should_stop():
            monitor.eval(sess)
            time.sleep(args.monitor_eval_secs)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def main(_):
    """
        Tensorflow for monitoring trained policy
    """

    args = HParams

    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Reset graph before allocating any?
    tf.reset_default_graph()

    if args.job_name == "monitor":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run_monitor(args, server)

if __name__ == "__main__":
    tf.app.run()
