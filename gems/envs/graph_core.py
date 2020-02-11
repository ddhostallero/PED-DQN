import numpy as np

NONE = 0
ASK = 1
GIVE = 2


class Agent(object):
    def __init__(self, n_neighbors=2):
        self.itype = "agent"
        
        self.n_neighbors = n_neighbors
        self.obs_dim = (self.n_neighbors + 1)*3
        self.act_spc = 3

        self._obs = None
        self.n_need = 0
        self.n_has = 0
        self.action = 0
        self.neighbors = []
        self.tx_eval = 0

    def give(self, receiver):
        if self.n_has > 0:
            receiver.n_has += 1
            self.n_has -= 1
    
    def get_obs(self):
        obs = [[self.n_has, self.n_need, self.action]]
        obs += [[x.n_has, x.n_need, x.action] for x in self.neighbors]
        obs = np.asarray(obs).reshape(3*(self.n_neighbors+1))

        return obs
        
    def collect_evals(self):
        self.collected_evals = np.asarray([x.tx_eval for x in self.neighbors]).mean()
        return self.collected_evals*beta

    def base_reward(self):
        return 0


class World(object):
    def __init__(self, n_agents):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.n_agents = n_agents

        self.adjacency_matrix = None
        self.adj_list = []
        self.step = 0

    def reset(self):
        self.step = 0
        
    # update state of the world
    def physical_step(self, action_n):
        self.step += 1
        
        for agent, action in zip(self.agents, action_n):
            agent.action = action

        for agent in self.agents:
            action = agent.action

            if action == NONE or action == ASK:
                continue
            elif action == GIVE:
                receiver = []
                for neighbor in agent.neighbors:
                    if neighbor.action == ASK:
                        receiver.append(neighbor)

                np.random.shuffle(receiver)
                for r in receiver:
                    agent.give(r)


    def incentive_step(self, inc_n):
        for i, agent in enumerate(self.agents):
            agent.tx_eval = inc_n[i]
