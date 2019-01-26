"""Arguments and hyper-parameters all in hand.
"""
import argparse

parser = argparse.ArgumentParser(description=None)

# SYSTEM
parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
parser.add_argument('--task', default=0, type=int, help='Task index')
parser.add_argument('--job-name', default="worker", help='worker or ps')
parser.add_argument('-w', '--num_workers', default=8, type=int, help='Number of workers')
parser.add_argument('-r', '--remotes', default=None, help='The address of pre-existing VNC servers and rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('-n', '--dry_run', action='store_true', default=False, help="Print out commands rather than executing them")
parser.add_argument('-m', '--mode', type=str, default='nohup', help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")

# PATHS
parser.add_argument('-l', '--logdir', default="convnet_stage2", help='Log directory path')
parser.add_argument('-o', '--outdir', default="inferences/conv2_results", help='output log directory. Default: none')

parser.add_argument('-pp','--log_accu',default='s0',help='the ID for log_accu files')
parser.add_argument('-xL','--system_size',default=16)
"""Do we need to distinguish stages?
"""

# TESTING
parser.add_argument("--render", action="store_true", help="Set true to rendering")
parser.add_argument("--num_tests", default=1000, type=int, help="Number of episodes to run.")
parser.add_argument("--monitor_eval_secs", type=int, default=20, help="Time interval of monitor evaluations.")

# TRAINING
parser.add_argument('-p', '--policy', type=str, default='cnn', help='Choose policy network: simple, cnn')
parser.add_argument('--num_training_steps', type=int, default=int(1e10), help='Number of global training steps.')
parser.add_argument('--local_steps', type=int, default=40, help='Local steps of updating the agent' )
#parser.add_argument('--timestep_limit', type=int, default=1024, help='Time step limit for timing out' )
parser.add_argument('--solver', type=str, default='adam', help='Type of optimizer: rmsprop, adam')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--grad_clip', type=float, default=40.0)
parser.add_argument('--gamma_factor', type=float, default=0.99, help='Discounted factor in MDP')
parser.add_argument('--entropy_cost', type=float, default=0.0001, help='Const of entropy loss')
parser.add_argument('--exploration', type=float, default=0.001, help='Exploration probability (not yet sched)')
parser.add_argument('--failure_reward', type=float, default=0.0, help='Punishment when update failure')
parser.add_argument('--accept_reward', type=float, default=1.0, help='Reward when update is accepted')
parser.add_argument('--strategy', type=str, default="egreedy", help='Option: egreedy, ea6')

# Env
parser.add_argument('--env_id', default="IcegameEnv-v0", help='Environment id')
# -- traning condition
parser.add_argument('--num_mcsteps', default=4000, type=int, help='Number of MC steps for SSF update')
parser.add_argument('--defect_upper_thres', default=2, type=int, help='Used in discrte_criterion() for setting thresholds')
parser.add_argument('--defect_lower_thres', default=6, type=int, help='Used in discrte_criterion() for setting thresholds')
parser.add_argument('--dconfig_amp', default=5, type=float)
parser.add_argument('--stepwise_invfactor', default=0.001, type=float)
parser.add_argument('--smallsize_discount', default=1.0, type=float)
parser.add_argument('--config_refresh_steps', default=10000, type=int)
#parser.add_argument('--disable_local', action='store_false', default=True)

# usage: from params import args
HParams = parser.parse_args()
