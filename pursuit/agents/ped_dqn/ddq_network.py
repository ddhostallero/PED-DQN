from agents.ind_dqn.dq_network import DQN as BaseDQN
import numpy as np
import tensorflow as tf
import config

FLAGS = config.flags.FLAGS

class DQN:
    def __init__(self, sess, state_dim, sup_state_dim, action_dim, n_agents, nn_id):
        self.sess = sess
        self.action = BaseDQN(sess, state_dim, sup_state_dim, action_dim, n_agents, nn_id)
        self.mission = BaseDQN(sess, state_dim, sup_state_dim, action_dim, n_agents, "lng"+nn_id, use_as_peer=True)

    def training_target_qnet(self):
        self.sess.run([self.action.update_slow_target_dqns, self.mission.update_slow_target_dqns])

    def training_peer_qnet(self):
        self.sess.run(self.mission.update_peer_dqns)

    def training_a_qnet(self, *params):
        return self.action.training_qnet(*params)

    def training_m_qnet(self, *params):
        return self.mission.training_qnet(*params)

    def get_aq_values(self, *params):
        return self.action.get_q_values(*params)

    def get_mq_values(self, *params):
        return self.mission.get_q_values(*params)

    def get_aq_mq_values(self, state_ph):
        return self.sess.run([self.action.concat_dqns, self.mission.concat_dqns],
            feed_dict={self.action.state_ph: state_ph,
                       self.mission.state_ph: state_ph})

    def get_aq_pmq_values(self, state_ph):
        return self.sess.run([self.action.concat_dqns, self.mission.concat_peer_dqns],
            feed_dict={self.action.state_ph: state_ph,
                       self.mission.next_state_ph: state_ph})


    def get_aq_tmq_values(self, state_ph):
        return self.sess.run([self.action.concat_dqns, self.mission.concat_target_dqns],
            feed_dict={self.action.state_ph: state_ph,
                       self.mission.next_state_ph: state_ph})
