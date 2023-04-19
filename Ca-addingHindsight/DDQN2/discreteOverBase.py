import gym
import numpy as np
from gym.spaces import Discrete


class DiscreteOverBase(gym.ActionWrapper):
    """
    
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(21)
    
    def action(self, act:int):
    #    print(( np.array([act],dtype=np.float32) -10)/10)
        return ( np.array([act],dtype=np.float32) -10)/10