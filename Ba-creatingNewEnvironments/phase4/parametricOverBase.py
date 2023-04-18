import gym
import numpy as np
from gym.spaces import Tuple, Box, Discrete


class ParametricOverBase(gym.ActionWrapper):
    """
    
    """

    def __init__(self, env):
        super().__init__(env)
        parameters_min = np.array([0, 0], dtype="float32")                      # min of charge with grid and PV
        parameters_max = np.array([1, 1], dtype="float32")                      # RELATIVE max of charge with grid and PV
        self.action_space = Tuple((Discrete(2), Box(parameters_min[0:1], parameters_max[0:1]), Box(parameters_min[1:], parameters_max[1:])))
    
    def action(self, act:Tuple):
        id, parameters0, parameters1 = act
        if id == 0:
            print(parameters0)
            return parameters0
        elif id == 1:
            print(parameters1)
            return -parameters1
        else: 
            raise ValueError('action not chosen from dicrete grid or pv')



