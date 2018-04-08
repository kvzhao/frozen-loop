"""Icegame Environment Version 3:
    * This version features using config autocorr as reward function each episode.
    * corresponds to libicegame2
"""
from __future__ import division

import gym
from gym import error, spaces, utils, core

#from six import StingIO
import sys, os
import json
import random
import numpy as np
from icegame import SQIceGame, INFO
import time
from collections import deque
from datetime import datetime

rnum = np.random.randint

"""TODO and Discussion List:
"""
#Q: Are these hyper-parameters needed to be here?
#Q: do we still need life? --> NO, lives is bad mechanism.
# For now, NO timeout NO lives
#Q: Do we only trust one ep counter?

# the following parameters are depreciated.
NUM_OBSERVATION_MAPS = 7 # in total
NUM_ACTIONS = 7

"""TODO List
    * Set reward function more clever --> Try to use autocorr as reward.
"""

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def transf_binary_vector(raw_state):
    """Transform from 2d to vector and binarization."""
    state = raw_state.flatten()
    state[state < 0.0] = -1.0
    state[state > 0.0] = +1.0
    return state

def autocorr(statevec, refvec):
    """Inner product of current and reference state.
      args:
        statevec (numpy array): current configuration
        ref (numpy array): referenced configuration
      returns:
        autocorr (float): value of correlation
    """
    # inner product of two vector
    corr = np.sum(statevec * refvec) / len(refvec)
    return corr

class IcegameEnv(core.Env):
    def __init__ (self, L, kT, J, 
                    stepwise_reward="constant",
                    end_reward="loopsize",
                    terminate_mode="trial",
                    obs_type="multi",
                ):
        """IceGame
            *** Considering more action and state spaces. Use autocorr as reward. ***
          Args:
            stepwise:
            endreward:
            terminate_mode:
                * metro: Each time metropolis is executed, then call it an episode.
                * trial: Finite trial times each episodes
            obs_type (observation type):
                * multi:
                * global_local:
            reset_each_epsidoes:
                reset configuration each # of episodes
        """
        self.L = L
        self.kT = kT
        self.J = J
        self.N = 4*L**2
        self.sL = int(np.sqrt(self.N)) # square length L 
        self.stepwis = stepwise_reward
        self.endreward = end_reward
        self.terminate_mode = terminate_mode
        self.obs_type = obs_type
        num_neighbors = 1
        num_replicas = 1
        num_mcsteps = 4000
        self.num_mcsteps = num_mcsteps
        num_bins = 1
        num_thermalization = num_mcsteps
        tempering_period = 1
        self.mc_info = INFO(self.L, self.N, num_neighbors, num_replicas, \
                num_bins, num_mcsteps, tempering_period, num_thermalization)

        self.sim = SQIceGame(self.mc_info)
        self.sim.set_temperature (self.kT)
        self.sim.init_model()
        self.sim.mc_run(num_mcsteps)

        self.episode_terminate = False
        self.accepted_episode = False

        self.last_update_step = 0
        # why do we need to keep track last returned results?
        self.last_rets = None

        # Extend the action to 8+1 = 9 actions
        self.idx2act = dict({
                            0 :   "head_0",
                            1 :   "head_1",
                            2 :   "head_2",
                            3 :   "tail_0",
                            4 :   "tail_1",
                            5 :   "tail_2",
                            6 :   "metropolis",
                            })

        self.act2idx  = {v: k for k, v in self.idx2act.items()}

        # action space and state space
        # global_observation_space
        self.observation_space = spaces.Box(low=-1, high=1.0,
            shape=(self.L, self.L, 4), dtype=np.float32)
        # local_observation_space
        self.local_observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(len(self.idx2act))
        self.reward_range = (-1, 1)

        # reference configuration: buffer for initial config each episode
        self.refconfig = None
        # TODO: Scheduling reward scale
        self.reward_scale = 5.0
        self.reward_threshold = 0.0 # 5 * (1/1024.0)
        self.reward_trajectory = []

        """Choose Observation Function
        """

        # output file
        self.ofilename = "loop_sites.log"
        # render file
        self.rfilename = "loop_renders.log"
        # save log to json for future analysis
        self.json_file = "env_history.json"
        self.env_settinglog_file = "env_settings.log"

        self.stacked_axis = 2

        ## counts reset()

    def auto_step(self):
        # auto_step works as long loop algorithm.
        guides = self.sim.guide_action()
        # Or, we can execute metropolis when guide fails
        E, D, dC = self.sim.get_phy_observables()
        if (E == -1):
            act = self.name_action_mapping["metropolis"]
        else:
            act = np.random.choice(guides)
        return self.step(act)

    def step(self, action):
        """step function
            Args: action

            TODO:
                Taking nested list of actions as a single 'action' on markov chain transition.
        """
        terminate = False
        reward = 0.0 # -0.000975 # stepwise punishment.
        obs = None
        info = None
        metropolis_executed = False

        ## execute different type of actions
        ## maybe we need a better action index
        if (action == 6):
            self.sim.flip_trajectory()
            rets = self.sim.metropolis()
            metropolis_executed = True
        elif (0 <= action < 6) :
            rets = self.sim.move(action)

        """ Results from icegame
            index 0 plays two roles:
                if action is walk:
                    rets[0] = is_icemove
                elif action is metropolis:
                    rets[0] = is_accept
        """
        is_accept, dEnergy, dConfig = rets
        is_icemove = True if is_accept > 0.0 else False
        self.last_rets = rets

        # metropolis judgement
        if (metropolis_executed):
            """TODO: Add autocorr of config here.
            """
            # compute the reward which for rejection
            if self.terminate_mode == "metro":
                terminate = True

            if is_accept > 0 and dConfig > 0:
                """ Updates Accepted
                    1. Calculate rewards
                      1.1 Get current configuration before updating
                      1.2 calculate the inner product
                      1.3 reward = 1.0 - autocorr
                    2. Save logs
                    3. Reset maps and buffers
                """
                #current_config = self._transf2d(self.sim.get_state_tp1_map_color())
                #statevec = transf_binary_vector(current_config)

                self.sim.update_config()
                print ("[GAME_ENV] PROPOSAL ACCEPTED!")

                total_steps = self.sim.get_total_steps()
                ep_steps = self.sim.get_ep_step_counter()
                ep = self.sim.get_episode()
                loop_length = self.sim.get_accepted_length()[-1]
                loop_area = self.calculate_area()

                # get counters
                action_counters = self.sim.get_action_statistics()
                metropolis_times = self.sim.get_updating_counter()
                update_times = self.sim.get_updated_counter()

                # compute update interval
                update_interval = total_steps - self.last_update_step
                self.last_update_step = total_steps

                # acceptance rate
                total_acc_rate = self.sim.get_total_acceptance_rate() * 100.0
                effort =  update_times/total_steps * 100.0
                # calculate the metropolis reward
                #acorr = autocorr(statevec, self.refconfig)
                #reward = (1.0 - acorr) * self.reward_scale
                reward = 1.0

                # TODO: Calculate recent # steps' acceptance rate
                """Dump resutls into file.
                """
                # output to self.ofilename
                with open(self.ofilename, "a") as f:
                    f.write("1D: {}, \n(2D: {})\n".format(self.sim.get_trajectory(), self.convert_1Dto2D(self.sim.get_trajectory())))
                    print ("\tSave loop configuration to file: {}".format(self.ofilename))

                print ("\tTotal accepted number = {}".format(update_times))
                print ("\tAccepted loop length = {}, area = {}".format(loop_length, loop_area))
                print ("\tAgent walks {} steps in episode, action counters: {}".format(ep_steps, self.sim.get_ep_action_counters()))
                action_stats = [x / total_steps for x in action_counters]
                print ("\tStatistics of actions all episodes (ep={}, steps={}) : {}".format(ep, total_steps, action_stats))
                print ("\tAcceptance ratio (accepted/ # of metropolis) = {}%".format(
                                                                    update_times * 100.0 / metropolis_times))
                print ("\tAcceptance ratio (from icegame) = {}%".format(total_acc_rate))
                print ("\tRunning Effort = {}%".format(effort))

                # TODO: How to describe the loop?
                info = {
                    "Acceptance Ratio" : total_acc_rate,
                    "Running Effort": effort,
                    "Updated" : update_times,
                    "Loop Size": loop_length,
                    "Loop Area": loop_area,
                }

                # Render when special case happened.
                if loop_area >= 1 or loop_length >= 8:
                    self.render()
                self.dump_env_status()
                self.sim.clear_buffer()

                """ Terminate?
                    stop after accpetance, will increase the episode rewards.
                    But can we still running the program to increase the total rewards?
                    Or do not terminate, just reset the location?
                """

                # reset the initial position and clear buffer
                # TODO: Check the difference
                # no need to reset here
                # self.sim.reset(rnum(self.N))

            else:
                self.sim.clear_buffer()
                reward = 0.0

                """
                    Rejection or dConfig <= 0
                        1. Keep updating with new canvas.
                            or
                        Early stop.
                        2. Wrong decision penalty
                    Q: Should we reset the initial location?
                    Q: How to handle no config change?
                """

            # reset or update
        else:
            """Stepwise feedback:
                1. exploration
                2. icemove reards
                3. defect propagation guiding
                4. #more

                TODO: Write option in init arguments.
            """
            # Check each scale (each of them stays in 0~1)
            reward = 0.0

        obs = self.get_obs()

        # Add the timeout counter (TODO: Check these codes)
        # Terminate and run monte carlo to prepare new config
        #terminate = True
        #print ("[GAME_ENV] Reset Ice Configuration!")
        #self.sim.reset_config()

        # Not always return info
        self.reward_trajectory.append(reward)
        return obs, reward, terminate, info

    # Start function used for agent learning
    def start(self, init_site=None, create_defect=True):
        """
            Q: Do we flip at start?
                I think flip @ starting point is reasonable.
            Returns: same as step()
                obs, reward, terminate, rets
        """
        if init_site == None:
            init_agent_site = self.sim.start(rnum(self.N))
        else:
            init_agent_site = self.sim.start(init_site)
        assert(self.agent_site == init_agent_site)
        if create_defect:
            self.sim.flip()
        self.episode_terminate = False
        # remove this legacy?

        state = self.get_obs()
        # reference configuration
        #self.refconfig = transf_binary_vector(state.configs_2d[:,:,0])

        return state

    def reset(self, site=None, create_defect=True):
        """reset is called by RL convention.
        """
        ## clear buffer and set new start of agent
        if site is None:
            site = rnum(self.N)
        init_site = self.sim.reset(site)
        assert(init_site == site)
        self.episode_terminate = False
        # actually, counter can be called by sim.get_episode()

        if create_defect:
            self.sim.flip()

        """TODO
            This mechanism should be checked.
            Reset configuration: run monte carlo again.
        """
        #print ("[GAME_ENV] Reset Ice Configuration!")
        #self.sim.reset_config()

        info = None
        self.last_rets = None
        self.reward_trajectory = []

        state = self.get_obs()
        # reference configuration
        # self.refconfig = transf_binary_vector(state.configs_2d[:,:,0])

        return state

    def timeout(self):
        return self.sim.timeout()

    @property
    def is_terminate(self):
        """Return whether game is terminate"""
        # use less
        return self.episode_terminate

    def get_game_status(self):
        """(TODO)Return the game status including steps and physical observables. 
          returns:
        """
        total_steps = self.sim.get_total_steps()
        ep_steps = self.sim.get_ep_step_counter()
        ep = self.sim.get_episode()

        # get counters
        metropolis_times = self.sim.get_updating_counter()
        update_times = self.sim.get_updated_counter()

        # compute update interval
        update_interval = total_steps - self.last_update_step

        # acceptance rate
        total_acc_rate = self.sim.get_total_acceptance_rate() * 100.0
        effort =  update_times/total_steps * 100.0

        d = {
            "total_steps": total_steps,
            "updated_times": update_times,
        }

        return AttrDict(d)

    def set_output_path(self, path):
        self.ofilename = os.path.join(path, self.ofilename)
        self.rfilename = os.path.join(path, self.rfilename)
        self.json_file = os.path.join(path, self.json_file)
        self.env_settinglog_file = os.path.join(path, self.env_settinglog_file)
        print ("Set environment logging to {}".format(self.ofilename))
        print ("Set loop and sites logging to {}".format(self.rfilename))
        print ("Set results dumpping path to {}".format(self.json_file))
        print ("Set env setting log path to {}".format(self.env_settinglog_file))

        self.save_env_settings()

    @property
    def agent_site(self):
        return self.sim.get_agent_site()

    @property
    def agent_site2d(self):
        #TODO FIX
        return (self.sim.get_agent_site()//self.sL, self.sim.get_agent_site()%self.sL)

    def set_agent_site(self, site, clear_map=False):
        #Notice: sim.start() is just set agent on site,
        #   but not clear the maps. (call restart if needed.)
        if 0 <= site < self.N:
            if clear_map:
                self.sim.restart(site)
            else:
                self.sim.start(site)

    @property
    def action_name_mapping(self):
        return self.idx2act

    @property
    def name_action_mapping(self):
        return self.act2idx

    @property
    def physical_observables(self):
        return self.sim.get_phy_observables()

    # TODO: Need to replace these codes.
    ## ray test  (for: int, list, np_list)
    def convert_1Dto2D(self, input_1D):
        """This function is provided by Thisray.
            The problematic function, fixing is needed.
        """
        output_2D = None
        if type(input_1D) == int:
            output_2D = (int(input_1D/self.L), int(input_1D%self.L))
        elif type(input_1D) == list:
            output_2D = []
            # better use of list comprehension
            for position in input_1D:
                output_2D.append((int(position/self.L), int(position%self.L)))
        return output_2D

    def calculate_area(self):
        """TODO:
            The periodic boundary condition can be modified.
            This function is provided by Thisray.
            The problematic function, fixing is needed.
        """
        traj_2D = self.convert_1Dto2D(self.sim.get_trajectory())
        traj_2D_dict = {}
        for x, y in traj_2D:
            if x in traj_2D_dict:
                traj_2D_dict[x].append(y)
            else:
                traj_2D_dict[x] = [y]

        # check Max y_length
        y_position_list = []
        for y_list in traj_2D_dict.values():
            y_position_list = y_position_list + y_list
        y_position_list = list(set(y_position_list))
        max_y_length = len(y_position_list) -1

        area = 0.0
        for x in traj_2D_dict:
            diff = max(traj_2D_dict[x]) - min(traj_2D_dict[x])
            if diff > max_y_length:
                diff = max_y_length
            temp_area = diff - len(traj_2D_dict[x]) +1  ## avoid vertical straight line
            if temp_area > 0:
                area = area + temp_area

        return area

    def set_ice(self, s):
        """Convert numpy array into python list, then set_ice"""
        if type(s) == np.ndarray:
            s = s.tolist()
            self.sim.set_ice(s)
        elif type(s) == list:
            self.sim.set_ice(s)
        else:
            raise ValueError("Only numpy array or list are accepted.")

    # TODO: Option of Render on terminal or File.
    def render(self, mapname ="traj", mode="ansi", close=False):
        #of = StringIO() if mode == "ansi" else sys.stdout
        #print ("Energy: {}, Defect: {}".format(self.sqice.cal_energy_diff(), self.sqice.cal_defect_density()))
        s = None
        # TODO: 
        if (mapname == "traj"):
            s = self._transf2d(self.sim.get_canvas_map())
        start = self.sim.get_agent_init_site()
        start = (int(start/self.sL), int(start%self.sL))
        s[start] = 3
        screen = "\r"
        screen += "\n\t"
        screen += "+" + self.L * "---" + "+\n"
        for i in range(self.L):
            screen += "\t|"
            for j in range(self.L):
                p = (i, j)
                spin = s[p]
                if spin == -1:
                    screen += " o "
                elif spin == +1:
                    screen += " * "
                elif spin == 0:
                    screen += "   "
                elif spin == +2:
                    screen += " @ "
                elif spin == -2:
                    screen += " O "
                elif spin == +3:
                    # starting point
                    screen += " x "
            screen += "|\n"
        screen += "\t+" + self.L * "---" + "+\n"
        #TODO: Add choice write to terminal or file
        #sys.stdout.write(screen)
        with open(self.rfilename, "a") as f:
            f.write("Episode: {}, global step = {}\n".format(self.sim.get_ep_step_counter(), self.sim.get_total_steps()))
            f.write("{}\n".format(screen))

    def _get_multiple_channel_obs(self):
        """
            Need more flexible in get_obs. There will may be config, sequence, scalar observed states.
            TODO: add np.nan_to_num() to prevent ill value

            Three kinds of states:
                1. configs_2d
                    * starting config
                    * current config
                2. configs_1d
                    * current config vector --> useless?
                3. mini-maps
                    * agent map
                    * valid action map --> looks confusing
                    * canvas
                    * eng map
                    * defect map
                4. non-spatial information
                    * local neighboring spins
                    * physical observables
                    not sure we should concatenate them or not

            return: the dict object
        """
        st = self.sim.get_state_t_map_color()
        stp1 = self.sim.get_state_tp1_map_color()
        config_t_map = self._transf2d(st)
        config_tp1_map = self._transf2d(stp1)

        config_t_vec = self._transf1d(st)
        config_tp1_vec = self._transf1d(stp1)

        config_stack = np.stack([
                        config_t_map,
                        config_tp1_map
                        ], axis=self.stacked_axis)

        agent_map = self._transf2d(self.sim.get_agent_map())
        valid_map = self._transf2d(self.sim.get_valid_action_map())
        canvas_map = self._transf2d(self.sim.get_canvas_map())
        energy_map = self._transf2d(self.sim.get_energy_map())
        defect_map = self._transf2d(self.sim.get_defect_map())

        minimap_stack = np.stack([
                        agent_map,
                        valid_map,
                        canvas_map,
                        energy_map,
                        defect_map
                        ], axis=self.stacked_axis)

        local_spins = self.sim.get_local_spins()
        phyical_obs = self.sim.get_phy_observables()
        # non-spatial information

        local = local_spins + phyical_obs

        d = {
                    "configs_1d": config_tp1_vec,
                    "configs_2d": config_stack,
                    "minimaps": minimap_stack,
                    "local": local,
                }
        return AttrDict(d)

    def _get_local_global_obs(self):
        """
            Two types of states:
                1. global-maps
                    * agent map
                    * valid action map
                    * canvas
                    * eng map
                    * defect map
                2. local info
                    * local neighboring spins
                    * physical observables

            return: the dict object with key 
                * global_map
                * local_info
        """
        config_t_map = self._transf2d(self.sim.get_state_t_map_color())
        config_tp1_map = self._transf2d(self.sim.get_state_tp1_map_color())
        agent_map = self._transf2d(self.sim.get_agent_map())
        canvas_map = self._transf2d(self.sim.get_canvas_map())

        global_map = np.stack([
            config_t_map,
            config_tp1_map,
            agent_map,
            canvas_map
        ], axis=self.stacked_axis)

        local_spins = self.sim.get_local_spins()
        phyical_obs = self.sim.get_phy_observables()
        # non-spatial information
        local = local_spins + phyical_obs

        return {
                    "global_map" : global_map,
                    "local_info" : local,
                }

    def get_obs(self):
        """Get Observation: Critical function of environments.
        """
        local_spins = self._transf1d(self.sim.get_local_spins())
        local_sites = self._transf1d(self.sim.get_local_sites())

        # E, D, dC: but these values are too small and close to 0 or 1
        phyobs = self._transf1d(self.sim.get_phy_observables())
        local_obs = np.concatenate((local_spins, phyobs), axis=0) 

        # global observation
        diff_map = self._transf2d(self.sim.get_state_diff_map())

        # return in terms of dict
        # TODO
        d = {
            "local_spins" : local_spins,
            "local_obs"   : local_obs,
            "local_sites" : local_sites,
            "global_dmap" : diff_map,
        }

        return AttrDict(d)

    @property
    def unwrapped(self):
        """Completely unwrap this env.
            Returns:
                gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def save_env_settings(self):
        print ("")
        # Write settings into the logfile, modified when setting function is called.
        with open(self.env_settinglog_file, "a") as f:
            # TODO: write new parameters.
            f.write("Launch time: {}\n".format(str(datetime.now())))
            f.write("Number of Observation: {}\n".format(NUM_OBSERVATION_MAPS))
            #f.write("Stepwise reward function: {}\n".format(self.stepwise))
            #f.write("Metropolis reward function: {}\n".format(self.endreward))

    def _transf2d(self, s):
        # add nan_to_num here?
        return np.array(s, dtype=np.float32).reshape([self.sL, self.sL])

    def _transf1d(self, s):
        # suppose originally we have one dim vector
        return np.array(s, dtype=np.float32)

    def _append_record(self, record):
        with open(self.json_file, "a") as f:
            json.dump(record, f)
            f.write(os.linesep)

    def env_status(self):
        """Save status into jsonfile.
            * carefully choose items to be saved.
        """
        # get current timestamp
        total_steps = self.sim.get_total_steps()
        ep = self.sim.get_episode()
        # agent walk # steps in this episode
        ep_step_counters = self.sim.get_ep_step_counter()
        trajectory = self.sim.get_trajectory()
        if self.sim.get_accepted_length():
            loop_length = self.sim.get_accepted_length()[-1]
        else :
            loop_length = 0
        enclosed_area = self.calculate_area()
        update_times = self.sim.get_updated_counter()
        action_counters = self.sim.get_action_statistics()
        action_stats = [x / total_steps for x in action_counters]

        start_site = self.sim.get_agent_init_site()
        acceptance = update_times * 100.0 / ep

        d = {
            "Episode": ep,
            "Steps"  : total_steps,
            "StartSite"  : start_site,
            "Trajectory": trajectory,
            "UpdateTimes": update_times,
            "AcceptanceRatio" : acceptance,
            "LoopLength": loop_length,
            "EnclosedArea": enclosed_area,
            "ActionStats" : action_stats
        }
        return AttrDict(d)

    def dump_env_status(self):
        d = self.env_status()
        self._append_record(d)