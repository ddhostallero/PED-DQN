#!/usr/bin/env python
# coding=utf8
import make_env
import agents
import config
import time

import random
import numpy as np
import tensorflow as tf

FLAGS = config.flags.FLAGS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == '__main__':

    start_time = time.time()
    set_seed(FLAGS.seed)

    env = make_env.make_env(FLAGS.scenario)
    logger = config.logger

    trainer = agents.load(FLAGS.agent+"/step_trainer.py").Trainer(env, logger=logger)
    training_length = FLAGS.max_global_step

    print FLAGS.agent, config.file_name, training_length

    # # start learning
    if FLAGS.train == True:
        trainer.learn(training_length, FLAGS.max_step_per_ep)
    else:
        trainer.test(FLAGS.max_ep, FLAGS.max_step_per_ep)

    print "TRAINING TIME (sec)", time.time() - start_time
    trainer.save_nn()