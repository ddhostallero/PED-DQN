import numpy as np
from envs.grid_core import World
from envs.scenarios.pursuit_base import Scenario as BaseScenario
from envs.scenarios.pursuit_base import Prey as BasePrey
from envs.scenarios.pursuit_base import Predator as Agent
import config

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
map_size = FLAGS.map_size

OBJECT_TO_IDX = config.OBJECT_TO_IDX
pred = OBJECT_TO_IDX['predator']

max_step_per_ep = FLAGS.max_step_per_ep

class Predator(Agent):
    def __init__(self, quota):
        super(Predator, self).__init__(obs_range=2)
        self.power = 100
        self.step = 0.0
        self.gathered = 0
        self.obs_dim += 3

    def get_obs(self):
        self.step += 1.0
        return np.append(np.array(self._obs).flatten(), \
            [self.power/100.0, self.step/max_step_per_ep, self.gathered/(1.0*n_prey)])

    def base_reward(self, capture, is_terminal):
        self.gathered += capture

        reward = 0

        if self.action.u != 2:
            self.power -= 1     # allows negative power

        if self.gathered == n_prey or is_terminal:    # if this is the last step, calculate power reward
            reward += max(self.power, 0)         # more battery = higher reward (if succeed)
            # print self.gathered, n_prey, is_terminal

        if capture > 0:
            reward += 20*capture
            # print self.power, capture, reward, self.id

        return reward

    def is_done(self):
        return self.power <= 0 or self.gathered == n_prey

    def reset(self):
        self.power = 100
        self.gathered = 0
        self.step = 0.0

class Prey(BasePrey):
    def __init__(self):
        super(Prey, self).__init__()
        self._consumer_mask = np.ones((3,3),dtype=np.int8)
        self._consumer_mask[1,1] = 0
        self.gathered = 0

    def update_obs(self, obs):
        self._obs = obs.encode()[:,:,0]
        id_encoding = obs.encode_ids()

        minimap = (self._obs == pred)
        self.captured = (np.sum(minimap*self._consumer_mask) > 2) # at least 3 agents
        self.consumers = id_encoding[np.nonzero(self._consumer_mask * id_encoding)]

class Scenario(BaseScenario):

    def make_world(self):
        world = World(width=map_size, height=map_size)

        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }

        # add predators
        for i in xrange(n_predator):
            agents.append(Predator(1 + 1*(i==0)))
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
            agent.reset()
            world.placeObj(agent)

        world.set_observations()

        # fill the history with current observation
        for i in self.atype_to_idx["predator"]:
            world.agents[i].fill_obs()

        self.step = 0

    def reward(self, agent, world):
        if agent == world.agents[0]:
            self.step += 1
            self.prey_captured = 0
            for i in self.atype_to_idx["prey"]:
                prey = world.agents[i]
                if prey.exists and prey.captured:
                    world.removeObj(prey)
                    self.prey_captured += 1

        return agent.base_reward(self.prey_captured, (self.step == max_step_per_ep))

    def done(self, agent, world):
        return agent.is_done()

    def info(self, agent, world):
        return agent.gathered