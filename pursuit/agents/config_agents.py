#!/usr/bin/env python
# coding=utf8

def config_agent(_flags):
    flags = _flags

    flags.DEFINE_string("agent", "ped_dqn", "Agent")
    flags.DEFINE_float("beta", 1.0, "Weight for incentives")
    
    flags.DEFINE_integer("max_global_step", 1000000, "Number of steps to train")
    
    flags.DEFINE_integer("max_step_per_ep", 100, "Maximum step per episode")
    flags.DEFINE_integer("test_interval", 4000, "Number of steps/episodes before testing")
    flags.DEFINE_integer("train_interval", 1, "Number of steps/episodes before testing")
    flags.DEFINE_integer("target_update", 5000, "Number of (current) DQN training updates before updating the target network")

    flags.DEFINE_integer("rb_capacity", 20000, "Size of the replay memory")
    flags.DEFINE_integer("mrb_capacity", 20000, "Size of the replay memory")
    flags.DEFINE_integer("minibatch_size", 32, "Minibatch size")

    flags.DEFINE_float("gamma", 0.99, "Discount factor")
    flags.DEFINE_boolean("load_nn", False, "Load nn from file or not")
    flags.DEFINE_string("nn_file", "", "The name of file for loading")
    
    flags.DEFINE_boolean("train", True, "Training or testing")

    flags.DEFINE_integer("max_quota", 4, "Quota of the odd man out (Not being used)")
    flags.DEFINE_float("lr", 1e-4, "DQN or Actor Learning rate")
    flags.DEFINE_float("mlr", 5e-4, "M-DQN Learning rate")

    flags.DEFINE_integer("explore", 250000, "Exploration denom")

def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    filename = "a-" + FLAGS.agent + "-tu-" + str(FLAGS.target_update) + "-mq-" + str(FLAGS.max_quota)  
    filename += "-rb-" + str(FLAGS.rb_capacity) + "-mb-" + str(FLAGS.minibatch_size) + "-lr-" + str(FLAGS.lr)
    return filename