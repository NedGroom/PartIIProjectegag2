



import main
import sys
import os
import numpy as np
from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight
import matplotlib.pyplot as plt
import torch
import gym


#main.main(mode='train', train_eps=240, warmup=15, bsize=10, saveload='runa', validate_every=30, validate_eps=30, epsilon=5000, seed=1234)

#main.main(mode='train', train_eps=600, warmup=15, bsize=40, saveload='runa', validate_every=40, validate_eps=30, epsilon=5000, seed=1234)
#main.main(mode='train', train_eps=600, warmup=15, bsize=10, saveload='runa', validate_every=40, validate_eps=30, epsilon=50000, seed=1234)
#main.main(mode='train', train_eps=600, warmup=15, bsize=40, saveload='runa', validate_every=40, validate_eps=30, epsilon=50000, seed=1234)

#main.main(mode='test', validate_steps=2, validate_eps=6)

main.main(mode='train', train_eps=2000, warmup=15, bsize=64, saveload='runa', validate_every=100, validate_eps=30, epsilon=5000, seed=1, loadscaling = 5)
main.main(mode='train', train_eps=2000, warmup=15, bsize=64, saveload='runa', validate_every=100, validate_eps=30, epsilon=5000, seed=2, loadscaling = 5)
#main.main(mode='train', train_eps=300, warmup=15, bsize=10, saveload='runa', validate_every=30, validate_eps=30, epsilon=5000, seed=1235, loadscaling = 5)
#main.main(mode='train', train_eps=300, warmup=15, bsize=10, saveload='runa', validate_every=30, validate_eps=30, epsilon=5000, seed=1236, loadscaling = 5)



