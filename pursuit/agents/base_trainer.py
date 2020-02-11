import config
import numpy as np

FLAGS = config.flags.FLAGS
n_predator =  FLAGS.n_predator
quota = FLAGS.max_quota

def stringify(arr, separator=", "):
    arr = ["%.2f"%(x) for x in arr]
    return separator.join(arr) #+ "\n"

class Trainer(object):
    def __init__(self):
        self.sess = None
        self.saver = None
        self.n_agents = None
        self.agents = None
        self.predator_singleton = None

    def save_nn(self):
        self.saver.save(self.sess, config.nn_filename)

    def get_actions(self, obs_n, epsilon=0.0):
        random = np.random.rand((n_predator)) < epsilon

        act_n = self.predator_singleton.act_multi(obs_n[:n_predator], random)
        act_n = act_n.tolist()

        for agent, obs in zip(self.agents[n_predator:], obs_n[n_predator:]):
            act = agent.act(obs)
            act_n.append(act)

        return act_n

    def test(self, max_ep, max_step_per_ep, max_steps=10000):
        if max_steps < max_step_per_ep:
            max_steps = max_global_steps

        total_b_reward_per_episode = np.zeros((max_ep, self.n_agents))
        total_captured_per_episode = np.zeros((max_ep, self.n_agents))
        success_rate_per_episode = np.zeros((max_ep, self.n_agents))
        remaining_battery = np.zeros((n_predator))

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
                done = done_n[:n_predator].all()

                total_b_reward_per_episode[ep] += reward_n

                if done: 
                    break

                obs_n = obs_n_next

            if "battery" in FLAGS.scenario:
                for i in range(n_predator):
                    remaining_battery[i] += obs_n_next[i][-3]

            total_captured_per_episode[ep] = info_n
            
            success_rate_per_episode[ep, :n_predator] = 1*(total_captured_per_episode[ep, :n_predator] >= quota)

            total_steps_per_episode[ep] = step+1

        mean_b_reward = total_b_reward_per_episode[:ep_finished].mean(axis=0)
        mean_captured = total_captured_per_episode[:ep_finished].mean(axis=0)
        success_rate = success_rate_per_episode[:ep_finished].mean(axis=0)
        mean_steps = total_steps_per_episode[:ep_finished].mean()
        remaining_battery = remaining_battery/ep_finished

        return mean_steps, mean_b_reward, mean_captured, success_rate, remaining_battery