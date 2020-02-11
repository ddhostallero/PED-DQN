#!/usr/bin/env python
# coding=utf8

def config_env(_flags):
    flags = _flags

    # Scenario
    flags.DEFINE_string("scenario", "battery_endless", "Scenario")
    flags.DEFINE_integer("n_predator", 4, "Number of predators")
    flags.DEFINE_integer("n_prey", 3, "Number of preys")
    flags.DEFINE_string("rwd_form", "picsq", "reward form")

    # Observation
    flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")

    # core
    flags.DEFINE_integer("map_size", 10, "Size of the map")


def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    filename = "sc-"+FLAGS.scenario+"-h-"+str(FLAGS.history_len)
    filename += "-m-"+str(FLAGS.map_size)+"x"+str(FLAGS.map_size)
    filename += "-Pp-"+str(FLAGS.n_predator)+"x"+str(FLAGS.n_prey)
    
    return filename