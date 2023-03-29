"""Copied from https://gymnasium.farama.org/tutorials/implementing_custom_wrappers/"""
import gym
from gym import Discrete


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    wrapped_env = DiscreteActions(
        env, [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
    )
    print(wrapped_env.action_space)  # Discrete(4)