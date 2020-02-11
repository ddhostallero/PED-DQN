import numpy as np

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!

class MultiAgentEnv():
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, rx_callback=None, tx_callback=None, pre_encode=False):

        self.world = world
        self.agents = self.world.agents
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.rx_callback = rx_callback
        self.tx_callback = tx_callback

        self.pre_encode = pre_encode

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.agent_precedence = []

        for agent in self.agents:
            self.agent_precedence.append(agent.itype)
            self.observation_space.append(agent.obs_dim)
            self.action_space.append(agent.act_spc)

        self.bin_encoding = None

    def get_agent_profile(self):
        agent_profile = {}

        for i, agent in enumerate(self.agents):
            if agent.itype in agent_profile:
                agent_profile[agent.itype]['n_agent'] += 1
                agent_profile[agent.itype]['idx'].append(i)
            else:
                agent_profile[agent.itype] = {
                    'n_agent': 1,
                    'idx': [i],
                    'act_spc': agent.act_spc,
                    'obs_dim': agent.obs_dim
                }

        return agent_profile

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        self.agents = self.world.agents
        self.world.physical_step(action_n)

        if self.pre_encode:
            self.bin_encoding = self.world.get_bin_encoding()

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))

        return obs_n, np.asarray(reward_n), np.asarray(done_n), info_n

    def incentivize(self, incentives_n):
        rx_inc_n = []

        self.world.incentive_step(incentives_n)
        for agent in self.agents:
            agent.assign_incentive()
        for agent in self.agents:
            rx_inc_n.append(self._get_received(agent))

        return rx_inc_n

    def get_expenses(self, incentives_n):
        tx_inc_n = []
        
        self.world.incentive_step(incentives_n)
        for agent in self.agents:
            tx_inc_n.append(self._get_expenses(agent))

        return tx_inc_n

    def reset(self, args=None):
        # reset world
        self.reset_callback(self.world, args)

        if self.pre_encode:
            self.bin_encoding = self.world.get_bin_encoding()

        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def _get_done(self, agent):
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        return self.reward_callback(agent, self.world)

    def _get_received(self, agent):
        if self.rx_callback == None:
            return 0
        return self.rx_callback(agent, self.world)

    def _get_expenses(self, agent):
        if self.tx_callback == None:
            return 0
        return self.tx_callback(agent, self.world)

    def get_full_encoding(self, encoding="binary"):
        if encoding == "binary":
            if self.bin_encoding is None:
                return self.world.get_bin_encoding()
            else:
                return self.bin_encoding
        elif encoding == "id":
            return self.world.get_id_encoding()
        else:
            return self.world.get_full_encoding()

    def get_coordinates(self, agent_idxs):
        x = []
        y = []
        for i in agent_idxs:
            x.append(self.agents[i]._x)
            y.append(self.agents[i]._y)

        return x, y