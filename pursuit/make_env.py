def make_env(scenario_name, benchmark=False, pre_encode=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
    '''
    from envs.environment import MultiAgentEnv
    import envs.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()

    # create multiagent environment
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, 
                               reward_callback=scenario.reward, 
                               observation_callback=scenario.observation,
                               done_callback=scenario.done,
                               info_callback=scenario.info,
                               rx_callback=scenario.received,
                               tx_callback=scenario.transmitted,
                               pre_encode=pre_encode)
    return env
