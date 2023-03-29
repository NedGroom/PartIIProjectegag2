



import DDQN_discrete
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym


DDQN_discrete.main(batch_size=20,num_sample_eps=6, measure_step=10, num_episodes=1000, saveload='runddqncartpole', seed=1, update_step=20, loadscaling=3000)
DDQN_discrete.main(batch_size=20,num_sample_eps=6, measure_step=10, num_episodes=1000, saveload='runddqncartpole', seed=2, update_step=20, loadscaling=3000)
DDQN_discrete.main(batch_size=20,num_sample_eps=6, measure_step=10, num_episodes=1000, saveload='runddqncartpole', seed=3, update_step=20, loadscaling=3000)
#DDQN_discrete.main(num_sample_eps=6, measure_step=100, num_episodes=2000, saveload='runddqnc', seed=2, update_step=50, loadscaling=3000)


