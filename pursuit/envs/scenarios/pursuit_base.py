import numpy as np
from collections import deque
from envs.grid_core import World
from envs.grid_core import CoreAgent as Agent
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
map_size = FLAGS.map_size
beta = FLAGS.beta
OBJECT_TO_IDX = config.OBJECT_TO_IDX
pred = OBJECT_TO_IDX['predator']

use_attention = (FLAGS.agent == "ped_daqn")
ind_eval = (FLAGS.agent == "ped_daqn_rev")

class Prey(Agent):
    def __init__(self):
        super(Prey, self).__init__(
            act_spc=5, itype="prey")

        self._movement_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], dtype=np.int8)

        self.captured = False

    def update_obs(self, obs):
        self._obs = obs.encode()[:,:,0]

        pred = OBJECT_TO_IDX['predator']
        minimap = (self._obs == pred)
        self.captured = (np.sum(minimap*self._movement_mask) > 1)

    def reset(self):
        self.exists = True

    def collect_incentives(self, *args):
        return 0

class Predator(Agent):
    def __init__(self, obs_range=2):
        super(Predator, self).__init__(
            act_spc=5, obs_range=obs_range, itype="predator")

        self.obs_dim = 3*self.obs_dim*FLAGS.history_len
        self._obs = deque(maxlen=FLAGS.history_len)
        self.grid = None
        self.silent = False

    def get_obs(self):
        return np.array(self._obs).flatten()

    def can_observe_prey(self):
        prey = OBJECT_TO_IDX["prey"] 
        return (self._obs[:,:,0] == prey).any()

    def update_obs(self, obs):
        self.grid = obs
        obs = obs.bin_encode()
        self._obs.append(np.array(obs, dtype=np.int8)) 
        
    def fill_obs(self):
        # fill the whole history with the current observation
        for i in range(FLAGS.history_len-1):
            self._obs.append(self._obs[-1])

    def collect_incentives(self, use_sum=False):
        if ind_eval:
            ret_val = self.collected_incentives
            self.collected_incentives = 0
            return ret_val

        enc = self.grid.encode()

        enc[self.obs_range, self.obs_range,1] = 0
        
        if use_attention:
            return enc[:,:,1:]
        
        mask = enc[:,:,1]
        self.collected_incentives = 0
        if (mask > 0).any():
            if use_sum:
                self.collected_incentives = enc[:,:,2][mask.nonzero()].sum()
            else:
                self.collected_incentives = enc[:,:,2][mask.nonzero()].mean()
                
        # return self.collected_incentives*(beta + np.log(mask.sum()+1))
        return self.collected_incentives*beta

    def assign_incentive(self):
        # side = (2*self.obs_range + 1)
        # mask = self._obs[-1][:side**2].reshape((side,side,3))[:,:,1:].sum(axis=2)
        # mask[self.obs_range, self.obs_range] = 0

        # inc = np.asarray(self.action.c)
        # inc_sum = inc.sum()

        # masked_inc = (inc*mask)
        # excess = inc_sum - masked_inc.sum()

        # masked_att_sum = mask.sum(axis=(2,3), keepdims=True) + (mask.sum(axis=(2,3), keepdims=True) == 0)
        # norm_att = mask/masked_att_sum

        if not ind_eval:
            return

        for j in range(0, self.grid.height):
            for i in range(0, self.grid.width):
                v = self.grid.get(i, j)
                if v == None:
                    continue
                if v.t_id == pred:
                    v.collected_incentives += self.action.c[j][i]
                    # v.collected_incentives += (self.action.c[j][i] + excess)
        self.action.c = np.asarray(self.action.c).sum()

    def base_reward(self, capture):
        reward = (self.action.u != 2)*-0.1 # penalty for moving
        reward += self.collided*-0.1       # penalty for collision

        if capture:
            return reward + 5

        return reward

class Scenario(BaseScenario):
    def __init__(self):
        self.prey_captured = False

    def make_world(self):
        world = World(width=map_size, height=map_size)

        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }

        # add predators
        for i in xrange(n_predator):
            agents.append(Predator())
            self.atype_to_idx["predator"].append(i)

        # add preys
        for i in xrange(n_prey):
            agents.append(Prey())
            self.atype_to_idx["prey"].append(n_predator + i)

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, args=None):
        world.empty_grid()

        # randomly place agent
        for agent in world.agents:
            world.placeObj(agent)

        world.set_observations()

        # fill the history with current observation
        for i in self.atype_to_idx["predator"]:
            world.agents[i].fill_obs()

        self.prey_captured = False

    def reward(self, agent, world):
        if agent == world.agents[0]:
            for i in self.atype_to_idx["prey"]:
                prey = world.agents[i]
                if prey.captured:
                    self.prey_captured = True
                    break

        return agent.base_reward(self.prey_captured)

    def observation(self, agent, world):
        return agent.get_obs()
        
    def done(self, agent, world):
        return self.prey_captured

    def info(self, agent, world):
        return None

    def received(self, agent, world):
        return agent.collect_incentives()

    def transmitted(self, agent, world):
        return agent.collect_incentives(True)