import gym
from gym.spaces import Discrete
import math
import numpy as np
import bauwerk


if __name__ == "__main__":

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    
    print(env.action_space)  # Box([-1.], [1.], (1,), float32) : this is 'relative'

    obs = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, info = env.step(action)  # include truncated output if in Gym0.26
