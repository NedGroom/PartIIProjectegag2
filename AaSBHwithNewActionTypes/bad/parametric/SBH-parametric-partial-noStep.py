
import gym
from gym.spaces import Discrete, Tuple, Box
import math
import numpy as np
import bauwerk


class ParametricActions(gym.Wrapper):

    pmax = 3700     # pmax = 1, for relative power action type
    GRID = 0        # action ID
    PV = 1          # action ID

    def __init__(self, env):
        super().__init__(env)

        parameters_min = np.array([0, 0], dtype="float32")                      # min of charge with grid and PV
        parameters_max = np.array([self.pmax, self.pmax], dtype="float32")      # max of charge with grid and PV
        self.action_space = Tuple((Discrete(2),
                                          Box(parameters_min, parameters_max)))     
        #self.reset()
        
       

if __name__ == "__main__":

    

    print('hi')
    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = ParametricActions(
        env
    )
    print(wrapped_env.action_space)  

    obs = wrapped_env.reset(seed=42)
    for _ in range(20):
        action = wrapped_env.action_space.sample()
        print(action)
        print("typee of action space: ")
        print(type(wrapped_env.action_space))
        obs, reward, terminated, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
