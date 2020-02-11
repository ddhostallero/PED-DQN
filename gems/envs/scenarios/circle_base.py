import numpy as np
from envs.graph_core import Agent as AgentBase
from envs.graph_core import World
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS

n_agents = FLAGS.n_agents
beta = FLAGS.beta
max_step_per_ep = FLAGS.max_step_per_ep

NONE = 0
ASK = 1
GIVE = 2

max_res = FLAGS.supply
max_dem = FLAGS.demand

class Agent(AgentBase):
    def __init__(self, n_neighbors=2):
        super(Agent, self).__init__(n_neighbors=n_neighbors)
        
        max_vals = 1./np.asarray([max_res, max_res, self.act_spc-1])
        self.normalizer = (np.ones((n_neighbors+1, 3))*max_vals).reshape(self.obs_dim)
        self.obs_dim += 1

    def get_obs(self, step):
        obs = [[self.n_has, self.n_need, self.action]]
        obs += [[x.n_has, x.n_need, x.action] for x in self.neighbors]
        obs = np.asarray(obs).reshape(3*(self.n_neighbors+1))*self.normalizer
        self._obs = np.concatenate((obs, [step*1.0/max_step_per_ep]))

        return self._obs

    def check_success(self):
        return (self.n_has >= self.n_need)

    def collect_evals(self):
        self.collected_evals = np.asarray([x.tx_eval for x in self.neighbors]).mean()
        return self.collected_evals*beta        

    def get_squared_error(self):
        return (self.n_has - self.n_need)**2

    def get_abs_error(self):
        return np.abs(self.n_has - self.n_need)


class Scenario(BaseScenario):
    def __init__(self):
        self.ep = 0

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

    def make_adj_mat(self, agents):
        adj_mat = np.zeros((n_agents, n_agents))

        for i in xrange(n_agents):
            adj_mat[i, i-1] = 1
            adj_mat[i, (i+1)%n_agents] = 1

            agents[i].neighbors.append(agents[i-1])
            agents[i].neighbors.append(agents[(i+1)%n_agents])

        return adj_mat

    def reset_world(self, world, args=None):
        has_list = np.zeros(n_agents)
        need_list = np.zeros(n_agents)        
     
        max_has = max_res

        if FLAGS.ue == True:
            max_has = max_res + ((self.ep % (n_agents+1)) - (n_agents//2))

        max_need = max_dem - n_agents

        self.ep += 1

        idx = range(n_agents)
        np.random.shuffle(idx)

        for i in idx[:-1]:
            has_list[i] = np.random.randint(max_has)
            need_list[i] = np.random.randint(max_need)

            max_has -= has_list[i]
            max_need -= need_list[i]

        # has and need must be equal
        has_list[idx[-1]] = max_has
        need_list[idx[-1]] = max_need

        for i, agent in enumerate(world.agents):
            agent.n_has = has_list[i]
            agent.n_need = need_list[i] + 1 # minimum 1 need per agent
            agent.action = 0

        world.reset()
            
    def reward(self, agent, world):
         raise NotImplementedError()

    def observation(self, agent, world):
        return agent.get_obs(world.step)
        
    def done(self, agent, world):
        return (world.step == max_step_per_ep) or self.success

    def info(self, agent, world):
        return [agent.check_success(), agent.get_squared_error()]

    def received(self, agent, world):
        return agent.collect_evals()

    def full_state(self, world):
        full_state = []
        for agent in world.agents:
            full_state += [agent.n_has*1.0/max_res, agent.n_need*1.0/max_res, agent.action/(agent.act_spc-1.0)]

        return full_state