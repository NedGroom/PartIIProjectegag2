



import DDQN_discrete
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym


DDQN_discrete.main(batch_size=32, num_sample_eps=6, measure_step=100, num_episodes=2000, saveload='runDDQNe', seed=1, update_step=50, loadscaling=3000)
DDQN_discrete.main(batch_size=32, num_sample_eps=6, measure_step=100, num_episodes=2000, saveload='runDDQNe', seed=2, update_step=50, loadscaling=3000)


