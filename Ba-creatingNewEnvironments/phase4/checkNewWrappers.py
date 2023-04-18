

# for use of these tests, uncomment the print statement in the wrapper showing 
# how the sample is processed before being passed into the base wrapper

import gym
from newBaseEnvWrapper import NewBaseEnvWrapper
from discreteOverBase import DiscreteOverBase
from parametricOverBase import ParametricOverBase

env_name = 'bauwerk/SolarBatteryHouse-v0'


def contTest():
	env = gym.make(env_name)
	env = NewBaseEnvWrapper(env, tolerance = 0.3)
	print(env.action_space)
	obs = env.reset()
	for _ in range(5):
		action = env.action_space.sample()
		print(action)
		obs, reward, terminated, _, info = env.step(action)  # include truncated output if in Gym0.26


def discTest():
	env = gym.make(env_name)
	env = NewBaseEnvWrapper(env, tolerance = 0.3)
	env = DiscreteOverBase(env)
	print(env.action_space)
	obs = env.reset()
	for _ in range(5):
		action = env.action_space.sample()
		print(action)
		obs, reward, terminated, _, info = env.step(action)  # include truncated output if in Gym0.26


def paraTest():
	env = gym.make(env_name)
	env = NewBaseEnvWrapper(env, tolerance = 0.3)
	env = ParametricOverBase(env)
	print(env.action_space)
	obs = env.reset()
	for _ in range(5):
		action = env.action_space.sample()
		print(action)
		obs, reward, terminated, _, info = env.step(action)  # include truncated output if in Gym0.26


contTest()
discTest()
paraTest()