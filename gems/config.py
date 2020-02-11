#!/usr/bin/env python
# coding=utf8

import tensorflow as tf
import logging
import time
import envs.config_env as config_env
import agents.config_agents as config_agent
import os

flags = tf.flags

config_env.config_env(flags)
config_agent.config_agent(flags)

flags.DEFINE_integer("seed", 1, "seed value")
flags.DEFINE_string("folder", "", "Folder for the log file")

if not os.path.exists("./results/logs"):
        os.makedirs("./results/logs")

if not os.path.exists("./results/nn"):
        os.makedirs("./results/nn")

if flags.FLAGS.folder != "":
    if not os.path.exists("./results/logs/"+flags.FLAGS.folder):
        os.makedirs("./results/logs/"+flags.FLAGS.folder)

# Make result file with given filename
now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
file_name = config_env.get_filename() + "-" + config_agent.get_filename()
file_name += "-seed-" + str(flags.FLAGS.seed) + "-" + s_time

logger = logging.getLogger('Logger')
logger.propagate = False
logger.setLevel(logging.INFO)

log_filename = "./results/logs/" + flags.FLAGS.folder + "/r-" + file_name + ".txt"
logger_fh = logging.FileHandler(log_filename)

logger_fm = logging.Formatter('%(asctime)s\t%(message)s')
logger_fh.setFormatter(logger_fm)
logger.addHandler(logger_fh)

nn_filename = "./results/nn/" + flags.FLAGS.folder + "/nn-" + file_name