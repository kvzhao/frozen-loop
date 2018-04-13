
import argparse

parser = argparse.ArgumentParser(description='Analyise history from Icegame environment.')
parser.add_argument('-l', '--logdir', type=str, default='ExpDir', help='Path to experiment folder.')
parser.add_argument('-o', '--output', type=str, default='results', help='Path to output folder.')
parser.add_argument('-n', '--num_samples', type=int, default=1, help='Number of sampled trajectory.')


args = parser.parse_args()