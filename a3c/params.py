"""Arguments and hyper-parameters all in hand.
"""
import argparse

parser = argparse.ArgumentParser(description=None)

# SYSTEM
parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
parser.add_argument('--task', default=0, type=int, help='Task index')
parser.add_argument('--job-name', default="worker", help='worker or ps')
parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')

# PATHS
parser.add_argument('--log-dir', default="/tmp/icegame", help='Log directory path')
parser.add_argument('--out-dir', default=None, help='output log directory. Default: none')

"""Do we need to distinguish stages?
"""

# TESTING
parser.add_argument("--render", action="store_true", help="Set true to rendering")
parser.add_argument("--num-tests", default=1000, type=int, help="Number of episodes to run.")

# TRAINING
parser.add_argument('--env-id', default="IcegameEnv-v0", help='Environment id')
parser.add_argument('-p', '--policy', type=str, default='cnn', help='Choose policy network: simple, cnn')

# Env

# Hyper-parameter



# usage: from params import args
args = parser.parse_args()