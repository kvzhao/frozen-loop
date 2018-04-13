from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import logging

from envs import create_icegame_env
import models
from worker import FastSaver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def inference(args):
    indir = os.path.join(args.log_dir, 'train')
    outdir = os.path.join(args.log_dir, 'player') if args.out_dir is None else args.out_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(indir + "/checkpoint", "r") as f:
        first_line = f.readline().strip()
        print ("first_line is : {}".format(first_line))
    ckpt = first_line.split(' ')[-1].split('/')[-1][:-1]
    ckpt = ckpt.split('-')[-1]
    ckpt = indir + '/model.ckpt-' + ckpt

    print ("ckpt: {}".format(ckpt))

    # define environment
    env = create_icegame_env(outdir, args.env_id)
    # define environment
    local_space = env.local_observation_space.n
    global_space = env.global_observation_space.shape
    action_space = env.action_space.n

    with tf.device("/cpu:0"):
        # define policy network
        with tf.variable_scope("global"):
            if args.policy == "simple":
                policy = models.SimplePolicy(global_space, local_space, action_space)
            elif args.policy == "cnn":
                policy = models.CNNPolicy(global_space, local_space, action_space)
            policy.global_step = tf.get_variable("global_step", [], 
                    tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
        # Variable names that start with "local" are not saved in checkpoints.
        variables_to_restore = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_all_op = tf.global_variables_initializer()

        saver = FastSaver(variables_to_restore)

        # print trainable variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        logger.info('Trainable vars:')
        for v in var_list:
            logger.info('  {} {}'.format(v.name, v.get_shape()))
        logger.info("Restored the trained model.")

        # summary of rewards
        action_writers = []
        summary_writer = tf.summary.FileWriter(outdir)

        """NOT so useful.
        for act_idx in range(action_space):
            action_writers.append(tf.summary.FileWriter(
                os.path.join(outdir, "action_{}".format(act_idx))
            ))
        """

        logger.info("Inference events directory: %s", outdir)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        with tf.Session() as sess:
            logger.info("Initializing all parameters.")
            sess.run(init_all_op)
            logger.info("Restoring trainable global parameters.")
            saver.restore(sess, ckpt)
            logger.info("Restored model was trained for %.2fM global steps", sess.run(policy.global_step)/1000000.)

            #last_features = policy.get_initial_features()  # reset lstm memory
            length = 0
            rewards = 0

            # For plotting
            if args.render:
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec

                plt.ion()
                fig = plt.figure(num=None, figsize=(8, 8), dpi=92, facecolor='w', edgecolor='k')

                gs1 = gridspec.GridSpec(3, 3)
                gs1.update(left=0.05, right=0.85, wspace=0.15)
                ax1 = plt.subplot(gs1[:-1, :])
                ax2 = plt.subplot(gs1[-1, :-1])
                ax3 = plt.subplot(gs1[-1, -1])

                ax1.set_title("IceGame (UpTimes: {})".format(env.sim.get_updated_counter()))

                ind = np.arange(action_space)
                width = 0.20
                #action_legends = ["Up", "Down", "Left", "Right", "NextUp", "NextDown", "Metropolis"]
                action_legends = ["head_0", "head_1", "head_2", "tail_0", "tail_1", "tail_2", "Metro"]

                steps_energies=[]

            for ep in range(args.num_tests):
                """TODO: policy sampling strategy
                    random, greedy and sampled policy.
                """

                last_state = env.reset()
                steps_rewards=[]
                steps_values=[]

                # running policy
                while True:
                    fetched = policy.act_inference(last_state)
                    #fetched = policy.act_inference(last_state, *last_features)
                    prob_action, action, value_ = fetched[0], fetched[1], fetched[2]
                    #prob_action, action, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                    # Greedy
                    #print ("Prob of actions: {}".format(prob_action))
                    stepAct = action.argmax()
                    state, reward, terminal, info = env.step(stepAct)

                    # update stats
                    length += 1
                    rewards += reward
                    last_state = state
                    #last_features = features

                    if info:
                        loopsize = info["Loop Size"]
                        looparea = info["Loop Area"]

                    """Animation for State and Actions
                        Show Energy Bar On Screen.
                    """
                    if args.render:
                        # save list for plotting
                        steps_rewards.append(rewards)
                        steps_values.append(value_)
                        energy, _, _  = env.physical_observables
                        steps_energies.append(energy)

                        ax2.clear()
                        ax2.bar(ind, prob_action)
                        ax2.set_xticks(ind + width / 2)
                        ax2.set_xticklabels(action_legends)

                        canvas = state.global_obs[:,:,0]
                        ax1.clear()
                        ax1.imshow(canvas, 'Reds', interpolation="None",  vmin=-1, vmax=1)
                        ax1.set_title("IceGame: (UpTimes: {})".format(env.sim.get_updated_counter()))

                        ax3.clear()
                        #ax3.plot(steps_rewards, linewidth=2)
                        ax3.plot(steps_energies, linewidth=2)

                        plt.pause(0.05)
                        #plt.clf()

                    # store summary
                    summary = tf.Summary()
                    summary.value.add(tag='ep_{}/reward'.format(ep), simple_value=reward)
                    summary.value.add(tag='ep_{}/netreward'.format(ep), simple_value=rewards)
                    summary.value.add(tag='ep_{}/value'.format(ep), simple_value=float(value_[0]))

                    if info:
                        summary.value.add(tag='ep_{}/loop_size'.format(ep), simple_value=loopsize)
                        summary.value.add(tag='ep_{}/loop_area'.format(ep), simple_value=looparea)

                    summary_writer.add_summary(summary, length)
                    summary_writer.flush()

                    summary = tf.Summary()
                    """
                    for ac_id in range(action_space):
                        summary.value.add(tag='action_prob', simple_value=float(prob_action[ac_id]))
                        action_writers[ac_id].add_summary(summary, length)
                        action_writers[ac_id].flush()
                    """

                    """TODO:
                        1. Need more concrete idea for playing the game when interfering.
                        2. Save these values for post processing.
                        3. We need penalty for timeout. --> Move timeout into env.
                    """
                    #if length >= 1024:
                    #    terminal = True
                    if terminal:
                        #last_features = policy.get_initial_features()  # reset lstm memory
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))

                        length = 0
                        rewards = 0
                        break

        logger.info('Finished %d true episodes.', args.num_tests)
        if args.render:
            plt.savefig("GameScene.png")
        logger.info("Save the last scene to GameScene.png")
        env.close()


def main(_):
    from params import args
    inference(args)

if __name__ == "__main__":
    tf.app.run()