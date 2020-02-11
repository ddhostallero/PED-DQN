#!/usr/bin/env python
# coding=utf8

def config_env(_flags):
    flags = _flags

    # Scenario
    flags.DEFINE_string("scenario", "circle_priv_indiv_pabs", "Scenario")
    flags.DEFINE_integer("n_agents", 8, "Number of agents")

    flags.DEFINE_integer("supply", 40, "Number of available resources")
    flags.DEFINE_integer("demand", 40, "Number of needed resources")
    flags.DEFINE_boolean("ue", False, "Unequal supply and demand")

def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    if FLAGS.supply < 0:
        FLAGS.supply = FLAGS.n_agents*2
    if FLAGS.demand < 0:
        FLAGS.demand = FLAGS.n_agents*2        

    filename = "sc-"+FLAGS.scenario
    filename += "-na-"+str(FLAGS.n_agents) + "-s-" + str(FLAGS.supply) + "-d-" + str(FLAGS.demand)
    
    return filename