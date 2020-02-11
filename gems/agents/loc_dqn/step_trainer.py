from agents.loc_dqn.agent import Agent
import numpy as np
import tensorflow as tf
import config
from collections import deque
from datetime import datetime

np.set_printoptions(precision=2)

FLAGS = config.flags.FLAGS
minibatch_size = FLAGS.minibatch_size
n_agents = FLAGS.n_agents
test_interval = FLAGS.test_interval
train_interval = FLAGS.train_interval

def stringify(arr, separator=", "):
    arr = ["%.2f"%(x) for x in arr]
    return separator.join(arr) #+ "\n"

class Trainer(object):
    def __init__(self, environment, logger, use_incentive=True):
        self.env = environment
        self.logger = logger
        self.n_agents = n_agents

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self._agent_profile = self.env.get_agent_profile()
        agent_precedence = self.env.agent_precedence
        
        self.agent_singleton = Agent(act_space=self._agent_profile["agent"]["act_spc"],
                              obs_space=self._agent_profile["agent"]["obs_dim"],
                              sess=self.sess, n_agents=n_agents,
                              name="agent")


        # intialize tf variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if FLAGS.load_nn:
            if FLAGS.nn_file == "":
                logger.error("No file for loading Neural Network parameter")
                exit()
            self.saver.restore(self.sess, FLAGS.nn_file)
        else:
            self.agent_singleton.sync_target()

    def save_nn(self):
        self.saver.save(self.sess, config.nn_filename)

    def get_actions(self, obs_n, epsilon=0.0):
        random = np.random.rand((n_agents)) < epsilon
        act_n = self.agent_singleton.act_multi(obs_n[:n_agents], random)
        return act_n

    def get_incentives(self, info_n):
        inc_n = self.agent_singleton.incentivize_multi(info_n)
        return inc_n

    def learn(self, max_global_steps, max_step_per_ep):
        epsilon = 1.0
        epsilon_dec = 1.0/(FLAGS.explore)
        epsilon_min = 0.1

        start_time = datetime.now()

        if max_global_steps % test_interval != 0:
            max_global_steps += test_interval - (max_global_steps % test_interval)

        steps_before_train = min(FLAGS.minibatch_size*4, FLAGS.rb_capacity)

        tds = []
        ep = 0
        global_step = 0
        rxs = []
        while global_step < max_global_steps:
            ep += 1
            obs_n = self.env.reset()

            for step in xrange(max_step_per_ep):
                global_step += 1

                # Get the action using epsilon-greedy policy
                act_n = self.get_actions(obs_n, epsilon)

                # Do the action and update observation
                obs_n_next, reward_n, done_n, _ = self.env.step(act_n)
                
                done = done_n.all()
                done_n[:n_agents] = done

                transition = [obs_n, act_n, reward_n, obs_n_next, done_n]

                # get incentives for the transition
                inc_n = self.get_incentives(transition)

                # apply incentives
                rx_inc_n = self.env.incentivize(inc_n)

                exp = transition + [rx_inc_n, inc_n]

                self.agent_singleton.add_to_memory(exp)

                if global_step > steps_before_train and global_step % train_interval == 0:
                    td, rx = self.agent_singleton.train(global_step>50000)
                    tds.append(td)   
                    rxs.append(rx)

                if global_step % test_interval == 0:
                    mean_steps, mean_b_reward, mean_i_reward, mean_o_reward, \
                        success_rate, global_success_rate, squared_error = self.test(80, max_step_per_ep)
                
                    time_diff = datetime.now() - start_time
                    start_time = datetime.now()

                    est = (max_global_steps - global_step)*time_diff/test_interval 
                    etd = est + start_time

                    print global_step, ep, "%0.2f"%(mean_steps), mean_b_reward, "%.2f"%(mean_b_reward.mean()), "%0.2f"%epsilon
                    print "estimated time remaining %02d:%02d (%02d:%02d)"%(est.seconds//3600,(est.seconds%3600)//60,etd.hour,etd.minute)
                
                    self.logger.info("%d\tsteps\t%0.2f" %(global_step, mean_steps))
                    self.logger.info("%d\tb_rwd\t%s" %(global_step, stringify(mean_b_reward,"\t")))
                    self.logger.info("%d\ti_rwd\t%s" %(global_step, stringify(mean_i_reward, "\t")))
                    self.logger.info("%d\to_rwd\t%s" %(global_step, stringify(mean_o_reward, "\t")))
                    self.logger.info("%d\tsuccs\t%s" %(global_step, stringify(success_rate, "\t")))
                    self.logger.info("%d\tgbsuc\t%s" %(global_step, global_success_rate))
                    self.logger.info("%d\tsqerr\t%s" %(global_step, stringify(squared_error, "\t")))

                    td = np.asarray(tds).mean(axis=0)
                    rx = np.asarray(rxs).mean()
                    self.logger.info("%d\ttd_er\t%s" %(global_step, stringify(td, "\t")))
                    self.logger.info("%d\trx_in\t%s" %(global_step, str(rx)))
                    tds = []
                    rxs = []

                if done or global_step == max_global_steps: 
                    break

                obs_n = obs_n_next
                epsilon = max(epsilon_min, epsilon - epsilon_dec)

    def test(self, max_ep, max_step_per_ep, max_steps=10000):
        if max_steps < max_step_per_ep:
            max_steps = max_global_steps

        total_b_reward_per_episode = np.zeros((max_ep, self.n_agents))
        success_rate_per_episode = np.zeros((max_ep, self.n_agents))
        squared_error_per_episode = np.zeros((max_ep, self.n_agents))
        global_success_rate_per_episode = np.zeros((max_ep))

        total_steps_per_episode = np.ones(max_ep)*max_step_per_ep

        global_step = 0
        ep_finished = max_ep
        for ep in xrange(max_ep):
            if global_step > max_steps:
                ep_finished = ep
                break

            obs_n = self.env.reset()

            for step in xrange(max_step_per_ep):
                global_step += 1

                act_n = self.get_actions(obs_n)

                obs_n_next, reward_n, done_n, info_n = self.env.step(act_n)
                done = done_n.all()

                total_b_reward_per_episode[ep] += reward_n

                if done: 
                    break

                obs_n = obs_n_next
            
            success_rate_per_episode[ep] = info_n[:,0]
            squared_error_per_episode[ep] = info_n[:,1]
            global_success_rate_per_episode[ep] = np.asarray(info_n).all()
            total_steps_per_episode[ep] = step+1

        mean_b_reward = total_b_reward_per_episode[:ep_finished].mean(axis=0)
        success_rate = success_rate_per_episode[:ep_finished].mean(axis=0)
        squared_error = squared_error_per_episode[:ep_finished].mean(axis=0)
        global_success_rate = global_success_rate_per_episode.mean()
        mean_steps = total_steps_per_episode[:ep_finished].mean()

        return mean_steps, mean_b_reward, mean_b_reward, mean_b_reward, \
            success_rate, global_success_rate, squared_error