from agents.ind_dqn.agent import Agent
from agents.base_trainer import Trainer as BaseTrainer
from agents.base_trainer import stringify
from agents.simple_agent import RunningAgent as NonLearningAgent
import numpy as np
import tensorflow as tf
import config
from datetime import datetime

np.set_printoptions(precision=2)

FLAGS = config.flags.FLAGS
minibatch_size = FLAGS.minibatch_size
n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
test_interval = FLAGS.test_interval
train_interval = FLAGS.train_interval

class Trainer(BaseTrainer):
    def __init__(self, environment, logger):
        self.env = environment
        self.logger = logger
        self.n_agents = n_predator + n_prey

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self._agent_profile = self.env.get_agent_profile()
        agent_precedence = self.env.agent_precedence
        
        self.predator_singleton = Agent(act_space=self._agent_profile["predator"]["act_spc"],
                                        obs_space=self._agent_profile["predator"]["obs_dim"],
                                        sess=self.sess, n_agents=n_predator,
                                        name="predator")

        self.agents = []
        for i, atype in enumerate(agent_precedence):
            if atype == "predator":
                agent = self.predator_singleton
            else:
                agent = NonLearningAgent(self._agent_profile[atype]["act_spc"])

            self.agents.append(agent)

        # intialize tf variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if FLAGS.load_nn:
            if FLAGS.nn_file == "":
                logger.error("No file for loading Neural Network parameter")
                exit()
            self.saver.restore(self.sess, FLAGS.nn_file)
        else:
            self.predator_singleton.sync_target()

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
        while global_step < max_global_steps:
            ep += 1
            obs_n = self.env.reset()

            for step in xrange(max_step_per_ep):
                global_step += 1                

                # Get the action using epsilon-greedy policy
                act_n = self.get_actions(obs_n, epsilon)

                # Do the action and update observation
                obs_n_next, reward_n, done_n, _ = self.env.step(act_n)
                # reward_n[:n_predator] = reward_n[:n_predator].mean()
                done = done_n[:n_predator].all()
                done_n[:n_predator] = done          

                exp = [obs_n[:n_predator], act_n[:n_predator], 
                       reward_n[:n_predator], obs_n_next[:n_predator], done_n[:n_predator]]
                self.predator_singleton.add_to_memory(exp)

                if global_step > steps_before_train and global_step % train_interval == 0:
                    td = self.predator_singleton.train()
                    tds.append(td)

                if global_step % test_interval == 0:
                    mean_steps, mean_b_reward, mean_captured, success_rate, rem_bat = self.test(25, max_step_per_ep)
                
                    time_diff = datetime.now() - start_time
                    start_time = datetime.now()

                    est = (max_global_steps - global_step)*time_diff/test_interval 
                    etd = est + start_time

                    print global_step, ep, "%0.2f"%(mean_steps), mean_b_reward[:n_predator], "%0.2f"%mean_b_reward[:n_predator].mean(), "%0.2f"%epsilon
                    print "estimated time remaining %02d:%02d (%02d:%02d)"%(est.seconds//3600,(est.seconds%3600)//60,etd.hour,etd.minute)
                
                    self.logger.info("%d\tsteps\t%0.2f" %(global_step, mean_steps))
                    self.logger.info("%d\tb_rwd\t%s" %(global_step, stringify(mean_b_reward[:n_predator],"\t")))
                    self.logger.info("%d\tcaptr\t%s" %(global_step, stringify(mean_captured[:n_predator], "\t")))
                    self.logger.info("%d\tsuccs\t%s" %(global_step, stringify(success_rate[:n_predator], "\t")))
                    self.logger.info("%d\tbttry\t%s" %(global_step, stringify(rem_bat, "\t")))

                    td = np.asarray(tds).mean(axis=1)
                    self.logger.info("%d\ttd_er\t%s" %(global_step, stringify(td[:n_predator], "\t")))
                    tds = []

                if done or global_step == max_global_steps: 
                    break

                obs_n = obs_n_next
                epsilon = max(epsilon_min, epsilon - epsilon_dec)

    