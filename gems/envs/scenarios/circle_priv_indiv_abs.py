import numpy as np
from envs.graph_core import World
from envs.scenarios.circle_base import Agent as AgentBase
from envs.scenarios.circle_indiv_abs import Scenario as BaseScenario
import config

FLAGS = config.flags.FLAGS

n_agents = FLAGS.n_agents
max_step_per_ep = FLAGS.max_step_per_ep

max_res = FLAGS.supply

class Agent(AgentBase):
    def __init__(self, n_neighbors=2):
        super(Agent, self).__init__(n_neighbors=n_neighbors)        
        self.obs_dim = 3 + self.n_neighbors
        
        self.normalizer = [max_res, max_res, self.act_spc-1]
        for i in range(n_neighbors):
            self.normalizer.append(self.act_spc-1)
        self.normalizer = 1./np.asarray(self.normalizer)
        
    def get_obs(self, step):
        obs = [self.n_has, self.n_need, self.action]
        [obs.append(x.action) for x in self.neighbors]
        self._obs = np.asarray(obs).reshape(3 + self.n_neighbors)*self.normalizer
        return self._obs

    def base_reward(self, terminal):
        return -1*self.get_abs_error()

class Scenario(BaseScenario):
    def make_world(self):
        world = World(n_agents=n_agents)

        agents = []

        # add agents
        for i in xrange(n_agents):
            agents.append(Agent(n_neighbors=2))

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1

        world.adjacency_matrix = self.make_adj_mat(agents)

        # make initial conditions
        self.reset_world(world)
        return world