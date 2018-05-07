"""
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
import distutils.version
from tflibs import *
from configs import hparams

def nipsHead(x):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 84, 84, 4]; output: [None, 2592] -> [None, 256];
    '''
    print('Using nips head design')
    x = tf.nn.relu(conv2d(x, 64, "l1", [3, 3], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l2", [3, 3], [2, 2], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, hparams.cell_size, "fc", normalized_columns_initializer(0.01)))
    return x

"""How make program switch policy easily?
"""

# Can we use RNN as the local info processor?

class SimplePolicy(object):
    """Simple single hidden layer forward network as the policy.
    """
    def __init__(self, global_space, local_space, ac_space, hparams):
        # local & global place holder
        print ('Use simple policy network')
        self.hparams = hparams
        self.ac_space = ac_space
        self.local_state = tf.placeholder(tf.float32, [None, local_space])
        # dummy placeholder
        self.global_state = tf.placeholder(tf.float32, [None] + list(global_space))

        feat = linear(self.local_state, 64, "feat", normalized_columns_initializer(0.01))

        self.logits = linear(feat, self.ac_space, "action", normalized_columns_initializer(0.01))
        self.vfunc = tf.reshape(linear(feat, 1, "value", normalized_columns_initializer(1)), shape=[-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, obs):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vfunc],
                        {self.local_state : [obs.local_obs]}) 

    def act_inference(self, obs):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vfunc],
                        {self.local_state : [obs.local_obs]}) 

    def value(self, obs):
        sess = tf.get_default_session()
        return sess.run(self.vfunc, {self.local_state : [obs.local_obs]})[0]

class CNNPolicy(object):
    """CNN Policy with two channels
    """
    def __init__(self, global_space, local_space, ac_space, hparams):
        # local & global place holder
        print ("Use two channels CNN policy")
        self.hparams = hparams
        self.ac_space = ac_space
        self.global_state = tf.placeholder(tf.float32, [None] + list(global_space))
        self.local_state = tf.placeholder(tf.float32, [None, local_space])

        """ Convolutional Channel """
        mconv1 = layers.conv2d(self.global_state,
                        num_outputs=8,
                        kernel_size=3,
                        stride=2,
                        activation_fn=tf.nn.relu,
                        scope="mconv1")
        mconv2 = layers.conv2d(mconv1,
                        num_outputs=16,
                        kernel_size=3,
                        stride=2,
                        activation_fn=tf.nn.relu,
                        scope="mconv2")

        l1 = linear(self.local_state, 32, "llayer1", normalized_columns_initializer(0.01))
        l2 = linear(l1, 64, "llayer2", normalized_columns_initializer(0.01))

        concat_feat = tf.concat([
            layers.flatten(mconv2),
            l2
        ], axis=1)

        #merge_layer
        merged = linear(concat_feat, 128, "merged",  normalized_columns_initializer(0.01))

        print ("Shape of mconv1: {}".format(mconv1.get_shape()))
        print ("Shape of mconv2: {}".format(mconv2.get_shape()))
        print ("Shape of ll1: {}".format(l1.get_shape()))
        print ("Shape of ll2: {}".format(l2.get_shape()))
        print ("Shape of concat: {}".format(concat_feat.get_shape()))
        print ("Shape of merged: {}".format(merged.get_shape()))

        self.logits = linear(merged, self.ac_space, "action", normalized_columns_initializer(0.01))
        self.vfunc = tf.reshape(linear(merged, 1, "value", normalized_columns_initializer(1)), shape=[-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, obs):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vfunc],
                        {self.local_state : [obs.local_obs],
                         self.global_state : [obs.global_obs]}) 

    def act_inference(self, obs):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vfunc],
                        {self.local_state : [obs.local_obs],
                         self.global_state: [obs.global_obs]}) 

    def value(self, obs):
        sess = tf.get_default_session()
        return sess.run(self.vfunc,
            {self.local_state : [obs.local_obs],
             self.global_state: [obs.global_obs]})[0]

class LSTMPolicy(object):
    def __init__(self, msize, ssize, isize, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        x = nipsHead(x)
        x = tf.expand_dims(x, [0])

        size = hparams.cell_size
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [obs], self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [obs], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [obs], self.state_in[0]: c, self.state_in[1]: h})[0]

def fullyconvHead(minimap, config, local):
    """Fully Convolutional Network, the naive one
        This head contains three input channels:
            * configuration channel
            * minimaps channel
            * local information channel
    """

    mconv1 = layers.conv2d(minimap,
                        num_outputs=32,
                        kernel_size=3,
                        stride=2,
                        activation_fn=tf.nn.relu,
                        scope="mconv1")
    mconv2 = layers.conv2d(mconv1,
                        num_outputs=64,
                        kernel_size=3,
                        stride=2,
                        activation_fn=tf.nn.relu,
                        scope="mconv2")

    cconv1 = layers.conv2d(config,
                        num_outputs=32,
                        kernel_size=3,
                        stride=2,
                        activation_fn=tf.nn.relu,
                        scope="cconv1")
    cconv2 = layers.conv2d(cconv1,
                        num_outputs=64,
                        kernel_size=3,
                        stride=2,
                        activation_fn=tf.nn.relu,
                        scope="cconv2")

    # why we need flatten?
    local_fc = layers.fully_connected(layers.flatten(local),
                        num_outputs=32,
                        activation_fn=tf.nn.tanh,
                        scope="local_fc")

    #NOTE: Not use spatial action for first trial
    #feat_conv = tf.concat([mconv2, cconv2], axis=3)

    feat_fc = tf.concat([
            layers.flatten(mconv2),
            layers.flatten(cconv2),
            local_fc
    ], axis=1)

    return feat_fc

class FConvLSTMPolicy(object):
    """Naive LSTM Policy with multiple input channels.
    TODO: NOT WORK YET, How to fix?
    """
    def __init__(self, mmap_space, config_space, local_space, ac_space):
        # Use more placeholder hanlding input data
        pass
        self.mmap_space = mmap_space
        self.config_space = config_space
        self.local_space = local_space
        self.ac_space = ac_space
        # Set inputs of networks
        self.minimap = tf.placeholder(tf.float32, [None] + list(mmap_space), name="minimap")
        self.configs = tf.placeholder(tf.float32, [None] + list(config_space), name="configs")
        self.local = tf.placeholder(tf.float32, [None, self.local_space], name="local")

        # Build head

        feat_fc = fullyconvHead(self.minimap, self.configs, self.local)
        """
            feat_fc --> projection --> LSTM --> output action (logits)
            TODO: projection layer is needed. (This part should port to another folder)
        """
        # LSTM Policy
        lstmsize = 256 # put it into hparams?
        lstm = rnn.BasicLSTMCell(lstmsize, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(feat_fc)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, feat_fc, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state

        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.minimap: [obs["minimaps"]], 
                        self.configs : [obs["configs"]],
                        self.local : [obs["local"]],
                            self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.minimap: [obs["minimaps"]], 
                        self.configs : [obs["configs"]],
                        self.local : [obs["local"]],
                            self.state_in[0]: c, self.state_in[1]: h})

    def value(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, 
                        {self.minimap: [obs["minimaps"]], 
                            self.configs : [obs["configs"]],
                            self.local : [obs["local"]],
                            self.state_in[0]: c, self.state_in[1]: h})[0]

class FConvPolicy(object):
    """Naive Feedforward Policy with multiple input channels.
        * Support two kinds of Head
    """
    def __init__(self, mmap_space, config_space, local_space, ac_space):
        # Use more placeholder hanlding input data
        pass
        self.mmap_space = mmap_space
        self.config_space = config_space
        self.local_space = local_space
        self.ac_space = ac_space
        # Set inputs of networks
        self.minimaps = tf.placeholder(tf.float32, [None] + list(mmap_space), name="minimap")
        self.configs = tf.placeholder(tf.float32, [None] + list(config_space), name="configs")
        self.local = tf.placeholder(tf.float32, [None, self.local_space], name="local")

        # Build head

        feat_fc = fullyconvHead(self.minimaps, self.configs, self.local)
        self.logits = linear(feat_fc, self.ac_space, "action", normalized_columns_initializer(0.01))

        self.vf = tf.reshape(linear(feat_fc, 1, "value", normalized_columns_initializer(1.0)), [-1])

        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def act(self, obs):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.minimaps: [obs["minimaps"]], 
                        self.configs : [obs["configs"]],
                        self.local : [obs["local"]] })

    def act_inference(self, obs):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf],
                        {self.minimaps: [obs["minimaps"]], 
                        self.configs : [obs["configs"]],
                        self.local : [obs["local"]]})

    def value(self, obs):
        sess = tf.get_default_session()
        return sess.run(self.vf, 
                        {self.minimaps: [obs["minimaps"]], 
                            self.configs : [obs["configs"]],
                            self.local : [obs["local"]]})[0]


def conv1x1_head(global_map, local_info):
    """This head will be designed to
        * SpatialMap --> conv1x1 --> action map (same size)
            * Now, dont use energy & defect map
        * local information channel

        GlobalMap = 2x Configs + Canvas + Agent map
        np.stack([
            configs,
            agent_map,
            canvas,
        ])

        ? Global map for walking action
        ? Local info for metropolis action
    """