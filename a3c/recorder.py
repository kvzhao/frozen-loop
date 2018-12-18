
import json
import numpy as np
from os.path import join

class PolicyRecorder(object):
  """Recoder is designed being independent of env_hist.
    * env_hist is recorded in episode while this class is written in step.
      actually, we analyze policy only in each episode.
    * Save all results no matter success or fail, in json format.

    NOTE: hist
      d = {
      "Episode": ep,
      "Steps"  : total_steps,
      "LocalSteps" : local_step,
      "StartSite"  : start_site,
      "Trajectory": trajectory,
      "UpdateTimes": update_times,
      "AcceptanceRatio" : acceptance,
      "LoopLength": loop_length,
      "EnclosedArea": enclosed_area,
      "ActionStats" : action_stats
    }
  """

  def __init__(self, path):
    """
      Args:
        path (str): Path to the specific folder for saving json files.
    """
    self.path = path # folder path

    self.current_episode=0
    self.is_accept = False

    # RL status
    self.steps=[]
    self.policy_probs=[]
    self.values=[]
    self.rewards = []
    self.actions = []

    # states
    self.local_spins=[]
    self.agent_spins=[]

    # physical quantities
    self.dconfigs = []
    self.denergies=[]
    self.energies = []


  def _clear(self):
    del self.steps[:] # equal to self.steps.clear()
    del self.policy_probs[:]
    del self.values[:]
    del self.local_spins[:]
    del self.agent_spins[:]
    del self.rewards[:]
    del self.energies[:]
    del self.denergies[:]
    del self.dconfigs[:]
    del self.actions[:]

  def attach_episode(self, epinum):
    """
      Use episode number as the file name.
      should we save init configuration?
    """
    self.current_episode =str(epinum)

  def push_step(self, step, act, pprobs, value, local, reward, accept=None):
    #TODO: record 'site'
    # accept is a boolean,
    # convert numpy array to list (pprobs)
    self.steps.append(step)
    self.policy_probs.append(pprobs)
    self.actions.append(act)
    self.values.append(value)
    self.rewards.append(reward)
    if accept is not None:
      self.is_accept = accept

    # parsing local state
    #self.local_states.append(local)
    localspin = local[:6]
    agent = local[6]
    eng, deng, dcfg = local[7:]
    self.local_spins.append(localspin)
    self.agent_spins.append(agent)
    self.dconfigs.append(dcfg)
    self.denergies.append(deng)
    self.energies.append(eng)

  def dump_episode(self):
    name = "ep_{}.json".format(self.current_episode)
    logfile = join(self.path, name)
    # file name denote P or F (pass or fail)
    # open the file and write results in? (with a for loop)
    # we may also save analysis results.

    with open(logfile, "w") as f:
      for step, a, pi, val, agent, spin, r, eng, deng, dcfg in zip(
        self.steps, self.actions, self.policy_probs, self.values,
        self.agent_spins, self.local_spins, self.rewards,
        self.energies, self.denergies, self.dconfigs):
        d = {
          "Step" : step,
          "Action" : a,
          "Policy" : pi,
          "Value" : val,
          "LocalSpin" : spin,
          "AgentSpin" : agent,
          "Reward" : r,
          "Energy" : eng,
          "DiffEng" : deng,
          "DiffConfig" : dcfg,
        }
        json.dump(d, f)
        f.write("\n")

    # clear memory of list.
    self._clear()
