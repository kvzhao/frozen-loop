"""Play with Icegame
    * enable/ disable subregion.
"""

from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import logging
sys.path.append("../icegame2/build/src")

import models
from envs import create_icegame_env
from worker import FastSaver
from recorder import PolicyRecorder
from khwutil import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





def inference(args):
    indir = os.path.join(args.logdir, 'train')
    outdir = os.path.join(args.logdir, 'player') if args.outdir is None else args.outdir
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
    #env = create_icegame_env(outdir, args.env_id, args)
    env = create_icegame_env(outdir, args.env_id)
    # define environment
    local_space = env.local_observation_space.n
    global_space = env.global_observation_space.shape
    action_space = env.action_space.n

    # resize the system and enable subregion
    if env.L != args.system_size:
        print ("Enlarge the system {} --> {}".format(env.L, int(args.system_size)))
        
        env.resize_ice_config(int(args.system_size), 10000)
        env.dump_env_setting()
        env.save_ice()

    print(env.L)
    Maper = BuildMaper(env.L)

    #exit(1)

    # our trained cnn always 32, 32
    env.enable_subregion()
    print ("Enable sub-region mechanism.")

    # policy recoder
    ppath = os.path.join(outdir, "episodes")
    if not os.path.exists(ppath):
        os.makedirs(ppath)
    pirec = PolicyRecorder(ppath)

    with tf.device("/cpu:0"):
        # define policy network
        with tf.variable_scope("global"):
            if args.policy == "simple":
                policy = models.SimplePolicy(global_space, local_space, action_space,args)
            elif args.policy == "cnn":
                policy = models.CNNPolicy(global_space, local_space, action_space,args)
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
                action_legends = ["head_0", "head_1", "head_2", "tail_0", "tail_1", "tail_2", "Metro"]

                steps_energies=[]
            

            Log_accuFwds = [[] for i in range(env.L**2*4)]
            Log_accuRevs = [[] for i in range(env.L**2*4)]
            Cx_dat = np.zeros(env.L) 
            Cy_dat = np.zeros(env.L)
            Sq_dat = np.zeros(env.L**2*4)
            Ncount = 0
            conf = None
            for ep in range(args.num_tests):
                """TODO: policy sampling strategy
                    random, greedy and sampled policy.
                """

                env.start(create_defect=True)

                if conf is None:
                    env.reset()
                    env.reset_ice_config()
                    env.sim.flip()
                    last_state = env.get_obs()
                    conf = env.sim.get_state_t()
                else:
                    env.reset() 
                    env.set_ice(conf)
                    env.sim.flip()
                    last_state = env.get_obs()
                
                loop_traj = []

                loop_traj.append(Maper[int(last_state['local_sites'][-1])])
                
                # these for plotting
                steps_rewards=[]
                steps_values=[]
                step = 0
                
                
                # policy recorder
                pirec.attach_episode(ep)
                # TODO: Call save_ice here?

                # running policy
                log_accuFwd_P = 0.
                log_accuRev_P = 0.
                while True:
                    fetched = policy.act_inference(last_state)
                    prob_action, action, value_ = fetched[0], fetched[1], fetched[2]

                    """TODO: Policy Recorder
                        * prob_action
                        * value_
                        * local config
                        * init_config (of course, but store in other way.)
                        * Store all cases
                        Q: Can we put these in env_hist.json?
                    """
                    ## deter :
                    #stepAct = action.argmax()

                    ## stochastic:
                    #print(prob_action)
                    stepAct = np.random.choice(7,1,p=prob_action)[0]
                    #print(stepAct)

                    state, reward, terminal, info = env.step(stepAct)
                    local = last_state.local_obs.tolist()
                    pi_ = prob_action.tolist()
                    value_ = value_.tolist()[0]
                    action_ = action.tolist()

                    # TODO: We need env 'weights', p(s, s', a) = ? (what the fuck is it?)
                    # And we also want some physical observables
                    pirec.push_step(step, stepAct, pi_, value_, local, reward)

                    # update stats
                    length += 1
                    step += 1
                    rewards += reward

                    fetched = policy.act_inference(state)
                    O_prob_action, O_action, O_value_ = fetched[0], fetched[1], fetched[2]
                    
                    PFwd = prob_action[stepAct]
                    PRev = O_prob_action[np.argwhere(state['local_sites']==last_state['local_sites'][-1])[0]][0]
                    log_accuFwd_P += np.log(PFwd)
                    log_accuRev_P += np.log(PRev) 
    
                    last_state = state
                    if stepAct != 6:
                        loop_traj.append(Maper[int(last_state['local_sites'][-1])])

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

                    """TODO:
                        1. Need more concrete idea for playing the game when interfering.
                        2. Save these values for post processing.
                        3. We need penalty for timeout. --> Move timeout into env.
                    """
                    if terminal:
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))
                        pirec.dump_episode()
                        length = 0
                        rewards = 0
                        step=0
                
                        ## check if loop is success to be constructed.
                        if info:
                            conf = env.sim.get_state_t() ## Update config
                            Cx,Cy = calc_corr(conf,env.L,Maper)
                            Sq    = calc_sq(conf,env.L)
                            Cx_dat += Cx/env.L**2/4
                            Cy_dat += Cy/env.L**2/4
                            Sq_dat += Sq
                            Ncount += 1
                            #print(Cx_dat)
                            #print(Cy_dat)
                            print(log_accuFwd_P,log_accuRev_P)
                            Log_accuFwds[loopsize].append(log_accuFwd_P)
                            Log_accuRevs[loopsize].append(log_accuRev_P)
                        break
                        
            for i in range(1024):
                np.save("log_accu/Fwd.%d.npd.%s"%(i,args.log_accu),np.array(Log_accuFwds[i]))
                np.save("log_accu/Rev.%d.npd.%s"%(i,args.log_accu),np.array(Log_accuRevs[i]))
            Cx_dat /= Ncount
            Cy_dat /= Ncount
            Sq_dat /= Ncount
            np.save("log_obs/Cx.npd.%s"%(args.log_accu),Cx_dat)
            np.save("log_obs/Cy.npd.%s"%(args.log_accu),Cy_dat)
            np.save("log_obs/Sq.npd.%s"%(args.log_accu),Sq_dat)
        logger.info('Finished %d true episodes.', args.num_tests)
        if args.render:
            plt.savefig("GameScene.png")
        logger.info("Save the last scene to GameScene.png")
        env.close()


def main(_):
    from params import HParams
    inference(HParams)

if __name__ == "__main__":
    tf.app.run()
