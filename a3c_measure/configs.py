"""This configs define hyper-parameters that used in network and algorithm.
    Save this configs together with model.
"""
import tensorflow as tf

# MDP
tf.app.flags.DEFINE_float("gamma", 0.9, "Discounted factor")

# model
tf.app.flags.DEFINE_string("head", "nips", "Define head of the input network")
tf.app.flags.DEFINE_integer("cell_size", 256, "Hidden units of LSTM policy")

# sovler
tf.app.flags.DEFINE_float("grad_clip", 40.0, "Gradient Clips")
tf.app.flags.DEFINE_float("learing_rate", 1e-4, "Initial learning rate")
tf.app.flags.DEFINE_integer("local_steps", 20, "Local steps of updating the agent")

hparams = tf.app.flags.FLAGS