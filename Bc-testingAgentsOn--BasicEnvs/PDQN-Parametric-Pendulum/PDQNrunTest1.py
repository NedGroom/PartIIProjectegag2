



import pdqn_copy_cut
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym


pdqn_copy_cut.run(episodes=2000, saveload='runPdqna250',num_sample_eps=6, loadscaling=3000, seed=1, measure_step=100)
pdqn_copy_cut.run(episodes=2000, saveload='runPdqna250',num_sample_eps=6, loadscaling=3000, seed=2, measure_step=100)

