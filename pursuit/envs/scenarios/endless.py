import numpy as np
from envs.grid_core import World
from envs.scenarios.battery_endless import Scenario as BaseScenario
from envs.scenarios.battery_endless import Prey as BasePrey
from envs.scenarios.battery_endless import Predator as Agent
import config

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
map_size = FLAGS.map_size

OBJECT_TO_IDX = config.OBJECT_TO_IDX
pred = OBJECT_TO_IDX['predator']

class Predator(Agent):
    def get_obs(self):
        self.step += 1.0
        return np.append(np.array(self._obs).flatten(), \
            [self.involved, self.step, self.gathered])

    def base_reward(self, capture, involved, is_terminal):
        self.gathered += capture
        self.involved += involved

        # reward = 0

        # if is_terminal:    # if this is the last step, calculate power reward
        #     reward += 25*self.gathered

        reward = capture*25

        return reward

class Prey(BasePrey):
    def update_obs(self, obs):
        self._obs = obs.encode()[:,:,0]
        id_encoding = obs.encode_ids()

        minimap = (self._obs == pred)
        self.captured = (np.sum(minimap*self._consumer_mask) == n_predator) # at least 3 agents
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