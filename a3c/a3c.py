from __future__ import print_function

import numpy as np
import tensorflow as tf
import six.moves.queue as queue
from collections import namedtuple
import scipy.signal
import threading
import distutils.version

from models import LSTMPolicy, FConvLSTMPolicy, FConvPolicy
import models
from env_runner import RunnerThread, process_rollout
from configs import hparams

class A3C(object):
    def __init__(self, env, args):
        """
            An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
            Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
            But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
            should be computed.

            Why not parsing args here? simplify the input arguments?
        """

        self.env = env
        hparams = self.hparams = args

        self.policy = args.policy
        self.task = args.task

        local_space = env.local_observation_space.n
        global_space = env.global_observation_space.shape
        action_space = env.action_space.n

        # environment information

        #TODO: How to use GPU?
        worker_device = "/job:worker/task:{}/cpu:0".format(self.task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                if self.policy == "simple":
                    self.network = models.SimplePolicy(global_space, local_space, action_space, self.hparams)
                elif self.policy == "cnn":
                    self.network = models.CNNPolicy(global_space, local_space, action_space, self.hparams)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                if self.policy == "simple":
                    self.local_net = pi = models.SimplePolicy(global_space, local_space, action_space, self.hparams)
                elif self.policy == "cnn":
                    self.local_net = pi = models.CNNPolicy(global_space, local_space, action_space, self.hparams)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None, ], name="adv")
            self.r = tf.placeholder(tf.float32, [None, ], name="r")

            #TODO: learning rate

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vfunc - self.r))
            # Note: What is the meaning of entropy here? Regularizer?
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            # batch size, this is not good
            bs = tf.to_float(tf.shape(pi.local_state)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * self.hparams.entropy_cost

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, self.hparams)

            grads = tf.gradients(self.loss, pi.var_list)

            #TODO: image summary with canvas: x[batch, :, :, 1], it seems maxout = 10
            #TODO: Add total rewards
            #TODO: Only save image when loop occurs
            #canvas = tf.expand_dims(pi.minimaps[0,:,:,2], axis=2)
            #canvas = tf.expand_dims(canvas, axis=0)

            """TODO:
                * Length of trajectories (or put in monitor)
            """
            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.scalar("model/entropy", entropy / bs)
            # tf.summary.image("model/canvas", canvas)
            #tf.summary.image("model/energy_map", energy_map)
            #tf.summary.image("model/defect_map", defect_map)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            self.summary_op = tf.summary.merge_all()

            grads, _ = tf.clip_by_global_norm(grads, self.hparams.grad_clip)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.local_state)[0])

            #TODO: LR decay
            # each worker has a different set of adam optimizer parameters

            if self.hparams.solver == "rmsprop":
                # 0.00025, 0.99, 0.0, 1e-6
                opt = tf.train.RMSPropOptimizer(self.hparams.learning_rate,
                                                decay=0.99, momentum=self.hparams.momentum, epsilon=1e-6)
            elif self.hparams.solver == "adam":
                opt = tf.train.AdamOptimizer(self.hparams.learning_rate)

            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
            self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
            process grabs a rollout that's been produced by the thread runner,
            and updates the parameters. The update is then sent to the parameter server.
        """

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()

        #TODO: hparams
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        #TODO: wired summary timing
        should_compute_summary = self.task == 0 and self.local_steps % 51 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        feed_dict = {
            self.local_net.local_state: batch.ls,
            self.local_net.global_state: batch.gs,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            #self.local_network.state_in[0]: batch.features[0],
            #self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
