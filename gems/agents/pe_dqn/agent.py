from agents.pe_dqn.ddq_network import DQN 
from agents.replay_buffer import ReplayBuffer
import numpy as np
import config

FLAGS = config.flags.FLAGS
rb_capacity = FLAGS.rb_capacity
minibatch_size = FLAGS.minibatch_size
target_update = FLAGS.target_update

discount_factor = FLAGS.gamma
mlr = FLAGS.mlr


# lta_t = 20000
# sta_t = 1000
class Agent(object):
    def __init__(self, obs_space, act_space, sess, n_agents, name):
        self.obs_space = obs_space
        self.act_space = act_space
        self.n_agents = n_agents

        self.pe_dqn = DQN(sess, obs_space, act_space, n_agents, name)

        self.rb = ReplayBuffer(capacity=rb_capacity)
        
        self.train_cnt = 0
        self.sns_q = None

    def reset(self):
        self.sns_q = None

    def act_multi(self, obs, random):
        if self.sns_q is None:
            q_values = self.pe_dqn.get_aq_values([obs])[0]
        else:
            q_values = self.sns_q
            
        r_action = np.random.randint(self.act_space, size=(len(obs)))
        action_n = ((random+1)%2)*(q_values.argmax(axis=1)) + (random)*r_action

        return action_n

    def incentivize_multi(self, info):
        state, action, reward, next_state, done = info
        done = done.all()

        # [ls_q, lns_q] = self.pe_dqn.get_mq_values([state, next_state])
        [[x, self.sns_q],[ls_q, lns_q]] = self.pe_dqn.get_aq_tmq_values([state, next_state])
        s_q = ls_q[range(self.n_agents), action]
        ns_q = discount_factor*lns_q.max(axis=1)*(not done) + reward

        td = ns_q - s_q    

        if done:
            self.sns_q = None

        return td

    def add_to_memory(self, exp):
        self.rb.add_to_memory(exp)

    def sync_target(self):
        self.pe_dqn.training_target_qnet()

    def train(self, use_rx):
        # train short DQN with recent data
        data = self.rb.sample_from_memory(minibatch_size)

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        base_reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])
        # tx_inc = np.asarray([x[6] for x in data])

        not_done = (done+1)%2

        if use_rx:
            rx_inc = np.asarray([x[5] for x in data])
            reward = base_reward + rx_inc
        else:
            rx_inc = np.zeros(1)
            reward = base_reward 

        td_error,_ = self.pe_dqn.training_a_qnet(state, action, reward, not_done, next_state)

        self.train_cnt += 1

        peer_update = False
        if self.train_cnt % target_update == 0:
            self.pe_dqn.training_target_qnet()
            self.pe_dqn.training_peer_qnet()
            peer_update = True

        return td_error, peer_update, np.abs(rx_inc).mean()