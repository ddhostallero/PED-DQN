import numpy as np
from envs.graph_core import World
from envs.scenarios.circle_priv_indiv_pabs import Scenario as BaseScenario
import config

FLAGS = config.flags.FLAGS

n_agents = FLAGS.n_agents
max_step_per_ep = FLAGS.max_step_per_ep

class Scenario(BaseScenario):
            
    def reward(self, agent, world):
        if agent == world.agents[0]:
            reward = 0
            for a in world.agents:
                reward += a.base_reward(False)

            self.reward = reward*1.0/n_agents # get mean

        return self.reward