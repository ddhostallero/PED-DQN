
# Inducing Cooperation through Reward Reshaping based on Peer Evaluations in Deep Multi-Agent Reinforcement Learning

### Primary author: David Earl Hostallero (david.hostallero@mail.mcgill.ca) 

## Overview
This is a **reimplementation** of the algorithms in the paper. The `gems` directory contains the simulator and algorithm source codes for the Resource Sharing environment. The `pursuit` directory contrains the simulator and algorithm source codes for the Paritally-Cooperative Pursuit.

## Running the program

### Resource Sharing

```
cd gems
python main.py --folder=gems --seed=1
```

### Partially-Coopertive Pursuit

```
cd pursuit
pythona main.py --folder=pursuit --seed=1
```

### Parameters

- `--seed`: the seed number for pseudo-random number generation
- `--folder`: subfolder where you want to save the logfiles and weights

## Disclaimer

This repository uses Python 2 and TensorFlow 1. Newer versions of the Python and TF may not be able to support this.