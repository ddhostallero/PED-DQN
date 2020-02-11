from agents.ped_dqn.ddq_network import DQN 
from agents.replay_buffer import ReplayBuffer
import numpy as np
import config

FLAGS = config.flags.FLAGS
rb_capacity = FLAGS.rb_capacity
mrb_capacity = FLAGS.mrb_capacity
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

        self.ped_dqn = DQN(sess, obs_space, act_space, n_agents, name)

        self.action_rb = ReplayBuffer(capacity=rb_capacity)
        self.mission_rb = ReplayBuffer(capacity=mrb_capacity)
        
        self.train_cnt = 0
        self.sns_q = None

        # self.lta = np.zeros((lta_t))
        # self.sta = np.zeros((sta_t))

    def reset(self):
        self.sns_q = None

    def act_multi(self, obs, random):
        if self.sns_q is None:
            q_values = self.ped_dqn.get_aq_values([obs])[0]
        else:
            q_values = self.sns_q
            
        r_action = np.random.randint(self.act_space, size=(len(obs)))
        action_n = ((random+1)%2)*(q_values.argmax(axis=1)) + (random)*r_action

        return action_n

    def incentivize_multi(self, info):
        state, action, reward, next_state, done = info
        done = done.all()

        # [ls_q, lns_q] = self.ped_dqn.get_mq_values([state, next_state])
        [[x, self.sns_q],[ls_q, lns_q]] = self.ped_dqn.get_aq_tmq_values([state, next_state])
        s_q = ls_q[range(self.n_agents), action]
        ns_q = discount_factor*lns_q.max(axis=1)*(not done) + reward

        td = ns_q - s_q    

        if done:
            self.sns_q = None

        return td

    def flush_action_rb(self):
        self.action_rb = ReplayBuffer(capacity=rb_capacity)

    def add_to_memory(self, exp):
        self.action_rb.add_to_memory(exp)
        self.mission_rb.add_to_memory(exp[:5])

    def sync_target(self):
        self.ped_dqn.training_target_qnet()

    def train_long_dqn(self):
        # train long DQN with recent and old data
        data = self.mission_rb.sample_from_memory(minibatch_size)

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        base_reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])

        not_done = (done+1)%2

        td_error,_ = self.ped_dqn.training_m_qnet(state, action, 
            base_reward, not_done, next_state, mlr)

        return td_error

    def train(self, use_rx):
        # train short DQN with recent data
        data = self.action_rb.sample_from_memory(minibatch_size)

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        base_reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])
        # tx_inc = np.asarray([x[6] for x in data])

        not_done = (done+1)%2

        if use_rx:
            rx_inc = np.asarray([x[5] for x in data])
            
            # ss = np.concatenate([state, next_state], axis=0)
            # lqv = self.ped_dqn.get_mq_values(ss)
            # ls_q = lqv[:minibatch_size]  # state
            # lns_q = lqv[minibatch_size:] # next state

            # mb_ind = range(minibatch_size)
            # for i in xrange(self.n_agents):
            #     s_q = ls_q[mb_ind, i, action[:,i]]
            #     ns_q = discount_factor*lns_q[:,i].max(axis=1)*(not_done[:,i]) + base_reward[:,i]
            #     tx_inc = ns_q - s_q

            #     inc = np.absolute(np.array([rx_inc[:,i], rx_inc[:,i] + tx_inc, rx_inc[:,i] - tx_inc]))
            #     rx_inc[:,i] = inc.min(axis=0)*np.sign(rx_inc[:,i])

            reward = base_reward + rx_inc
        else:
            rx_inc = np.zeros(1)
            reward = base_reward 

        td_error,_ = self.ped_dqn.training_a_qnet(state, action, reward, not_done, next_state)

        # tdmean = np.asarray(td_error).mean()
        # self.sta[self.train_cnt%sta_t] = tdmean
        # self.lta[self.train_cnt%lta_t] = tdmean

        self.train_cnt += 1

        peer_update = False
        if self.train_cnt % target_update == 0:
            self.ped_dqn.training_target_qnet()
            self.ped_dqn.training_peer_qnet()
            peer_update = True
        # if self.train_cnt % (target_update*2) == 0:
            # self.ped_dqn.action.training_target_qnet()
        # if self.train_cnt % target_update == 0:
            # self.ped_dqn.mission.training_target_qnet()
        # if self.train_cnt % (target_update*5) == 0:

        # if self.train_cnt % (target_update) == 0:
        #     self.ped_dqn.training_target_qnet()

        # if self.train_cnt == 50000:
        #     print '---updated peer---'
        #     self.ped_dqn.training_peer_qnet()
        #     peer_update = True

        # if self.train_cnt > lta_t and self.sta.mean() < self.lta.mean() - 1.96*self.lta.std()/np.sqrt(lta_t):
        #     print '---updated peer---'
        #     self.ped_dqn.training_peer_qnet()
        #     peer_update = True

        return td_error, peer_update, np.abs(rx_inc).mean()