from agents.ind_dqn.dq_network import DQN 
from agents.replay_buffer import ReplayBuffer
import numpy as np
import config

FLAGS = config.flags.FLAGS
rb_capacity = FLAGS.rb_capacity
minibatch_size = FLAGS.minibatch_size
target_update = FLAGS.target_update

if "battery" in FLAGS.scenario:
    sup_len = 3
elif "endless" in FLAGS.scenario:
    sup_len = 3
else:
    sup_len = 0

class Agent(object):
    def __init__(self, obs_space, act_space, sess, n_agents, name):
        self.obs_space = obs_space
        self.act_space = act_space
        self.n_agents = n_agents

        self.dqn = DQN(sess, obs_space, sup_len, act_space, n_agents, name)

        self.rb = ReplayBuffer(capacity=rb_capacity)       
        self.train_cnt = 0

    def act_multi(self, obs, random):            
        q_values = self.dqn.get_q_values([obs])[0]
        r_action = np.random.randint(self.act_space, size=(len(obs)))
        action_n = ((random+1)%2)*(q_values.argmax(axis=1)) + (random)*r_action

        return action_n

    def add_to_memory(self, exp):
        self.rb.add_to_memory(exp)

    def sync_target(self):
        self.dqn.training_target_qnet()

    def train(self):
        data = self.rb.sample_from_memory(minibatch_size)

        state = np.asarray([x[0] for x in data])
        action = np.asarray([x[1] for x in data])
        reward = np.asarray([x[2] for x in data])
        next_state = np.asarray([x[3] for x in data])
        done = np.asarray([x[4] for x in data])
        
        not_done = (done+1)%2

        td_error,_ = self.dqn.training_qnet(state, action, reward, not_done, next_state)

        self.train_cnt += 1
        if self.train_cnt % target_update == 0:
            self.dqn.training_target_qnet()

        return td_error