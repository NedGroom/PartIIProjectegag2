# this has been labelled bad and stored for reference, because an example i was drawing inspiration from (the action space example here:https://gymnasium.farama.org/tutorials/implementing_custom_wrappers/) 
# lead me to believe that I should process the action in the 'def action():' function in the wrapper, however this means that in the step function you cannot properly check that the action is contained in the action space.
# I could have kept the processing here, and checked validity of the 'def action():' output in step function in a different way, but I considered it more rigorous to specifically check it is contained in the action space, 
# as opposed to checking it is contained in a specifically designed set of results that 'seemed' to align to what i wanted.

# throws assertion error, because preprocesses the action away from the action_space type.

"""inspired from https://gymnasium.farama.org/tutorials/implementing_custom_wrappers/"""


import gym
from gym.spaces import Discrete
import math
import numpy as np
import bauwerk


class DiscreteActions(gym.ActionWrapper):

    pmax = 1     # pmax = 1, for relative power action type

    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(21)
    def action(self, act):
        print("run")
        return self.disc_to_cont(act,self.pmax)


def mydisc_to_cont_fun(action,pmax):
    action -= 10
    if action < -10:
        action = -10
    if action > 10:
        action = 10
    return  action/10 * pmax


def mydisc_to_cont_fun_array(action,pmax):
    action -= 10
    if action < -10:
        action = -10
    if action > 10:
        action = 10
    if action < 0:
        return np.array([abs(action)*pmax/10, 0])
    else:
        return np.array([0, action*pmax/10])


if __name__ == "__main__":

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = DiscreteActions(
        env, mydisc_to_cont_fun
    )
    print(wrapped_env.action_space)  # Discrete(21)

    obs = wrapped_env.reset(seed=42)
    for _ in range(1000):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
