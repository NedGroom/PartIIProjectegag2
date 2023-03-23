
# this has been labelled bad and stored for reference, because an example i was drawing inspiration from (the action space example here:https://gymnasium.farama.org/tutorials/implementing_custom_wrappers/) 
# lead me to believe that I should process the action in the 'def action():' function in the wrapper, however this means that in the step function you cannot properly check that the action is contained in the action space.
# I could have kept the processing here, and checked validity of the 'def action():' output in step function in a different way, but I considered it more rigorous to specifically check it is contained in the action space, 
# as opposed to checking it is contained in a specifically designed set of results that 'seemed' to align to what i wanted.

import gym
from gym.spaces import Discrete, Tuple, Box
import math
import numpy as np
import bauwerk


class ParametricActions(gym.ActionWrapper):

    pmax = 3700     # pmax = 1, for relative power action type
    GRID = 0        # action ID
    PV = 1          # action ID

    def __init__(self, env):
        super().__init__(env)

        parameters_min = np.array([0, 0], dtype="float32")                      # min of charge with grid and PV
        parameters_max = np.array([self.pmax, self.pmax], dtype="float32")      # max of charge with grid and PV
        self.action_space = Tuple((Discrete(2),
                                          Box(parameters_min, parameters_max)))     

    def action(self, action):
        print(action)
        print("processed")
        id, parameters = action
        if id == self.GRID:
            return -parameters[0]
        elif id == self.PV:
            return parameters[1]
        else: 
            raise ValueError('action not chosen from dicrete grid or pv')



if __name__ == "__main__":

    
    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = ParametricActions(
        env
    )
    print(wrapped_env.action_space)  

    obs = wrapped_env.reset(seed=42)
    for _ in range(20):
        action = wrapped_env.action_space.sample()
        print(action)
        print("startstep")
        obs, reward, terminated, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
        print("endloop")