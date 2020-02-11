import numpy as np
import tensorflow as tf
import config

FLAGS = config.flags.FLAGS

gamma = FLAGS.gamma  # reward discount factor
lr = FLAGS.lr    # learning rate
h_nodes = 32
m_nodes = 64

class DQN:
    def __init__(self, sess, global_state_dim, state_dim, action_dim, n_agents, nn_id):
        self.sess = sess
        self.global_state_dim = global_state_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        scope = 'dqn_' + str(nn_id)

        # placeholders
        self.global_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, global_state_dim], name='gs_ph')
        self.next_global_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, global_state_dim], name='gns_ph')

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])

        self.action_ph = tf.placeholder(dtype=tf.int32, shape=[None, n_agents])

        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.lr = tf.placeholder(dtype=tf.float32)

        # self.n_in = tf.placeholder(dtype=tf.int32)

        with tf.variable_scope(scope):
            self.ind_dqns = []
            q_values = []
            for i in range(n_agents):
                
                ind_dqn = self.generate_indiv_dqn(self.state_ph[:,i], i)
                self.ind_dqns.append(ind_dqn)

                # get the q-value of the state-action performed
                a_onehot = tf.one_hot(self.action_ph[:,i], action_dim, 1.0, 0.0)
                action_qval = tf.reshape(tf.reduce_sum(tf.multiply(ind_dqn, a_onehot), axis=1), (-1,1))
                q_values.append(action_qval)

            # use this for convenience in agent's act function
            self.concat_dqns = tf.reshape(tf.concat(self.ind_dqns, 1), (-1, self.n_agents, self.action_dim))

            q_values = tf.concat(q_values, axis=1)
            self.q_total = self.generate_mixing_network(self.global_state_ph, q_values)

        with tf.variable_scope('slow_target_'+scope):
            self.ind_target_dqns = []
            next_q_values = []
            for i in range(n_agents):
                ind_target_dqn = self.generate_indiv_dqn(self.next_state_ph[:,i], i, trainable=False)
                self.ind_target_dqns.append(ind_target_dqn)
                
                max_qval_next = tf.reshape(tf.reduce_max(ind_target_dqn, axis=1), (-1,1))
                next_q_values.append(max_qval_next)

            next_q_values = tf.concat(next_q_values, axis=1)
            self.slow_q_total = self.generate_mixing_network(self.next_global_state_ph, next_q_values, trainable=False)

        q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        target_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_'+scope)

        discount = self.is_not_terminal_ph * gamma
        target = self.reward_ph + discount * self.slow_q_total
         
        self.td_errors = tf.reduce_sum(tf.square(target - self.q_total))
        self.train_network = tf.train.AdamOptimizer(self.lr).minimize(self.td_errors, var_list=q_network_vars)

        # copy weights from q_network to target q_network
        update_slow_target_ops = []
        for i in range(len(q_network_vars)):
            assign_op = tf.assign(target_q_network_vars[i], q_network_vars[i])
            update_slow_target_ops.append(assign_op)
        self.update_slow_target_dqn = tf.group(*update_slow_target_ops)

    def generate_indiv_dqn(self, s, a_id, trainable=True):

        hidden = tf.layers.dense(s, h_nodes, activation=tf.nn.relu,
                             use_bias=True, reuse=tf.AUTO_REUSE, 
                             trainable=trainable, name='dense_a1_' + str(a_id))

        hidden2 = tf.layers.dense(hidden, h_nodes, activation=tf.nn.relu,
                             use_bias=True, reuse=tf.AUTO_REUSE, 
                             trainable=trainable, name='dense_a2_' + str(a_id))

        q_values = tf.layers.dense(hidden2, self.action_dim, reuse=tf.AUTO_REUSE,
                             trainable=trainable, name='q_value_' + str(a_id))

        return q_values

    def generate_mixing_network(self, global_state, q_values, trainable=True):
        hyper_in = global_state

        # For mixing network layer 1 (linear)
        w1 = tf.layers.dense(hyper_in, self.n_agents*m_nodes, 
            use_bias=True, trainable=trainable, name='dense_w1')

        # For mixing network layer 2
        w2 = tf.layers.dense(hyper_in, m_nodes,
            use_bias=True, trainable=trainable, name='dense_w2')

        # For mixing network hidden layer (linear)
        b1 = tf.layers.dense(hyper_in, m_nodes, #no activation 
            use_bias=True, trainable=trainable, name='dense_b1')

        # For mixing network output layer (2-layer hypernetwork with ReLU)
        b2_h = tf.layers.dense(hyper_in, m_nodes, activation=tf.nn.relu, 
            use_bias=True, trainable=trainable, name='dense_b2_h')

        b2 = tf.layers.dense(b2_h, 1, #no activation 
            use_bias=True, trainable=trainable, name='dense_b2')

        w1 = tf.reshape(tf.abs(w1), [-1, self.n_agents, m_nodes])
        w2 = tf.reshape(tf.abs(w2), [-1, m_nodes, 1])

        q_values = tf.reshape(q_values, [-1,1,self.n_agents])
        q_hidden = tf.nn.elu(tf.reshape(tf.matmul(q_values, w1),[-1,m_nodes]) + b1)
        q_hidden = tf.reshape(q_hidden, [-1,1,m_nodes])
        q_total = tf.reshape(tf.matmul(q_hidden, w2),[-1,1]) + b2

        return q_total

    def get_q_values(self, state_ph):
        return self.sess.run(self.concat_dqns, 
            feed_dict={self.state_ph: state_ph})#, self.n_in: len(state_ph)})

    def training_qnet(self, global_state_ph, state_ph, action_ph, reward_ph, 
        is_not_terminal_ph, next_global_state_ph, next_state_ph, lr=lr):
        
        return self.sess.run([self.td_errors, self.train_network], 
            feed_dict={
                self.global_state_ph: global_state_ph,
                self.state_ph: state_ph,
                self.next_global_state_ph: next_global_state_ph,
                self.next_state_ph: next_state_ph,
                self.action_ph: action_ph,
                self.reward_ph: reward_ph,
                self.is_not_terminal_ph: is_not_terminal_ph,
                # self.n_in: len(coords_ph),
                self.lr: lr})

    def training_target_qnet(self):
        self.sess.run(self.update_slow_target_dqn)