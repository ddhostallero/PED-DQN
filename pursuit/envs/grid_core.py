import numpy as np
import config
import sys

OBJECT_TO_IDX = config.OBJECT_TO_IDX

N = 0
E = 1
O = 2
W = 3
S = 4

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self, itype):
        assert itype in OBJECT_TO_IDX, itype
        self.itype = itype
        self.t_id = OBJECT_TO_IDX[itype]

        # name 
        self.name = ''
        # entity collides with others
        self.collide = True
        # exists in the map
        self.exists = True

    @property
    def pos(self):
        return self._x, self._y

    def set_pos(self, x, y):
        self._x = x
        self._y = y

# properties of agent entities
class CoreAgent(Entity):
    def __init__(self, act_spc=5, obs_range=1, name='', itype='agent'):
        super(CoreAgent, self).__init__(itype)
        
        self.name = name

        # cannot send communication signals
        self.silent = True

        # observation
        self.obs_range = obs_range
        self.obs_dim = (self.obs_range*2 + 1)**2
        
        # action
        self.act_spc = act_spc
        self.action = Action()

        # if waiting for other agents action
        self.waiting = False
        # if done doing its action in the current step
        self.done_moving = False
        # if the intended step collided 
        self.collided = False

        self._obs = None
        self._x = 0
        self._y = 0

        self.collected_incentives = 0
        self.power = 1.0

    def update_obs(self, obs):
        self._obs = obs

    def get_obs(self):
        return self._obs

    def base_reward(self, *args):
        return 0

    def reset(self):
        return

    def is_done(self):
        return False

    def assign_incentive(self):
        return

class Wall(Entity):
    def __init__(self):
        super(Wall, self).__init__('wall')

class Grid(object):
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 2
        assert height >= 2

        self.width = width
        self.height = height
        self.reset()
        self.singleton_wall = Wall()

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        if ((i >= 0 and i < self.width) and \
            (j >= 0 and j < self.height)):
            return self.grid[j * self.width + i]

        return self.singleton_wall

    def reset(self):
        self.grid = [None] * self.width * self.height

    def setHorzWall(self, x, y, length=None):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, self.singleton_wall)

    def setVertWall(self, x, y, length=None):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, self.singleton_wall)

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                v = self.get(x, y)
                # if x >= 0 and x < self.width and \
                #    y >= 0 and y < self.height:
                #     v = self.get(x, y)
                # else:
                #     v = self.singleton_wall

                grid.set(i, j, v)

        return grid

    def encode(self):
        """
        Produce a compact numpy encoding of the grid
        """

        array = np.zeros(shape=(self.height, self.width, 3))

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)
                if v == None:
                    continue

                array[j, i, 0] = v.t_id # entity type ID
                if isinstance(v, CoreAgent) and not v.silent and v.action.c is not None:
                    array[j, i, 1] = 1
                    array[j, i, 2] = v.action.c

        return array

    def bin_encode(self):
        """
        Produce a compact numpy encoding of the grid
        """

        array = np.zeros(shape=(self.height, self.width, 3))

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)
                if v == None:
                    continue

                array[j, i, v.t_id -1] = 1

        return array

    def encode_ids(self):
        """
        Produce a compact numpy encoding of the agent IDs
        """

        array = np.zeros(shape=(self.height, self.width), dtype=np.int8)

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)
                if isinstance(v, CoreAgent):
                    array[j, i] = v.id

        return array

# multi-agent world
class World(object):
    def __init__(self, width, height):
        # list of agents and entities (can change at execution-time!)
        self.agents = []

        self.dim_p = 2

        self.width = width
        self.height = height

        self.grid = Grid(self.width, self.height)
        self.step_cnt = 0

    def empty_grid(self):
        self.step_cnt = 0
        self.grid.reset()

    def placeObj(self, obj, top=(0, 0), size=None):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to randomly place
        """

        if size is None:
            size = (self.grid.width, self.grid.height)

        while True:
            x = np.random.randint(top[0], top[0] + size[0])
            y = np.random.randint(top[1], top[1] + size[1])

            # Don't place the object on top of another object
            if self.grid.get(x, y) != None:
                continue
            else:
                break

        self.grid.set(x, y, obj)
        obj.set_pos(x, y)
        return (x, y)

    def removeObj(self, obj):
        self.grid.set(obj._x, obj._y, None)
        obj.exists = False
        obj._x = -1
        obj._y = -1

    def single_agent_step(self, agent):
        if agent.done_moving or agent.waiting:
            return

        x, y = agent.pos
        action = agent.action.u

        if   action == N:
            y -= 1
        elif action == E:
            x -= 1
        elif action == W:
            x += 1
        elif action == S:
            y += 1
        elif action == O:
            agent.done_moving = True
            agent.collided = False
            return
        else:
            print "Action not recognized:", action
            sys.exit(1)

        intended_cell = self.grid.get(x, y)
        if isinstance(intended_cell, CoreAgent):
            agent.waiting = True
            # let the other agent move first
            self.single_agent_step(intended_cell)
            agent.waiting = False
            # get the intended cell (to check if it is empty)
            intended_cell = self.grid.get(x, y)

        # check if the intended cell is empty
        if not (intended_cell is None):  
            agent.collided = True
        else:
            x_0, y_0 = agent.pos
            self.grid.set(x_0, y_0, None)
            self.grid.set(x, y, agent)
            agent.set_pos(x, y)
            agent.collided = False

        agent.done_moving = True

    # update state of the world
    def physical_step(self, action_n):
        self.step_cnt += 1
        # set the action
        for i, agent in enumerate(self.agents):
            # agent cannot move if it doesn't exist or doesn't have power
            agent.done_moving = (not agent.exists) or (agent.power <= 0)
            agent.action.u = action_n[i]
            
        # do the action
        for agent in self.agents:
            self.single_agent_step(agent)

        # update observations of all agents
        self.set_observations()

    def incentive_step(self, inc_n):

        for i, agent in enumerate(self.agents):
            agent.action.c = inc_n[i]

    def set_observations(self):
        for agent in self.agents:
            if not agent.exists: continue

            x, y = agent.pos
            r = agent.obs_range
            obs = self.grid.slice(x-r, y-r,r*2+1,r*2+1)
            agent.update_obs(obs)

    def get_full_encoding(self):
        return self.grid.encode()

    def get_id_encoding(self):
        return self.grid.encode_ids()

    def get_bin_encoding(self):
        return self.grid.bin_encode()