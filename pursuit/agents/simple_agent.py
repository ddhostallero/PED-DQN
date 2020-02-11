import numpy as np

class DefaultAgent(object):
    def __init__(self, default_inc=0):
        self._default_inc = default_inc

    def act(self, obs, random=False):
        raise NotImplementedError()

    def incentivize(self, info):
        return self._default_inc

    def train(self):
        return 

    def add_to_memory(self, experience):
        return

    def sync_target(self):
        return

class RunningAgent(DefaultAgent):
    def __init__(self, action_dim, default_inc=0):
        self._action_dim = action_dim
        self._default_inc = default_inc

    def act(self, obs, random=False):
        # guided action (no collision)
        minimap = obs

        valid_act = []
        center = 1
        if minimap[center-1,center] == 0: # up
            valid_act.append(0)
        if minimap[center,center-1] == 0: #left
            valid_act.append(1)
        if minimap[center,center+1] == 0: #right
            valid_act.append(3)
        if minimap[center+1,center] == 0: #down
            valid_act.append(4)

        if len(valid_act) == 0:
            return 2

        return np.random.choice(valid_act)

class RandomAgent(DefaultAgent):
    def __init__(self, action_dim, default_inc=0):
        self._action_dim = action_dim
        self._default_inc = default_inc

    def act(self, obs, random=False):
        return np.random.randint(self._action_dim)

class StaticAgent(DefaultAgent):
    def __init__(self, action, default_inc=0):
        self._action = action
        self._default_inc = default_inc

    def act(self, obs, random=False):
        return self._action

    