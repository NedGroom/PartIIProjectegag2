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
import copy
from bauwerk.constants import (
    GYM_COMPAT_MODE,
    GYM_NEW_RESET_API_ACTIVE,
    GYM_RESET_INFO_DEFAULT,
    GYM_NEW_STEP_API_ACTIVE,
)


class DiscreteActions(gym.Wrapper):

    pmax = 1     # pmax = 1, for relative power action type

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(21)

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        info = {}

        action -= 10
        if action < -10:
            action = -10
        elif action > 10:
            action = 10
        action = action/10 * self.pmax
        
        print(action)

        action = float(action)  # getting the float value
        self.logger.debug("step - action: %1.3f", action)

        # Get old state
        load = self.state["load"]
        pv_generation = self.state["pv_gen"]
        cum_load = self.state["cum_load"]
        cum_pv_gen = self.state["cum_pv_gen"]

        action = self.get_power_from_action(action)
        attempted_action = copy.copy(action)

        if not self.cfg.grid_charging:
            # If charging from grid not enabled, limit charging to solar generation
            action = np.minimum(action, pv_generation)

        charging_power = self.battery.charge(power=action)

        # Get the net load after accounting for power stream of battery and PV
        net_load = load + charging_power - pv_generation

        if not self.grid.selling_allowed:
            net_load = np.maximum(net_load, 0)

        self.logger.debug("step - net load: %s", net_load)

        # Draw remaining net load from grid and get price paid
        cost = self.grid.draw_power(power=net_load)

        reward = -cost

        # Add impossible control penalty to cost
        info["power_diff"] = np.abs(charging_power - float(attempted_action))
        if self.cfg.infeasible_control_penalty:
            reward -= info["power_diff"]
            self.logger.debug(
                "step - cost: %6.3f, power_diff: %6.3f", cost, info["power_diff"]
            )

        # Get load and PV generation for next time step
        new_load = self.load.get_next_load()
        load_change = load - new_load
        load = new_load

        new_pv_generation = self.solar.get_next_generation()
        pv_change = pv_generation - new_pv_generation
        pv_generation = new_pv_generation

        battery_cont = self.battery.get_energy_content()

        cum_load += load
        cum_pv_gen += pv_generation
        self.time_step += 1

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_generation], dtype=self.cfg.dtype),
            "battery_cont": np.array(battery_cont, dtype=self.cfg.dtype),
            "time_step": int(self.time_step),
            "time_step_cont": self.time_step.astype(self.cfg.dtype),
            "cum_load": cum_load,
            "cum_pv_gen": cum_pv_gen,
            "load_change": np.array([load_change], dtype=self.cfg.dtype),
            "pv_change": np.array([pv_change], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(step=self.time_step),
        }

        observation = self._get_obs_from_state(self.state)

        terminated = bool(self.time_step >= self.cfg.episode_len)

        info["net_load"] = net_load
        info["charging_power"] = charging_power
        info["load"] = self.state["load"]
        info["pv_gen"] = self.state["pv_gen"]
        info["cost"] = cost
        info["battery_cont"] = battery_cont
        info["time_step"] = int(self.time_step)

        info = {**info, **self.grid.get_info()}

        self.logger.debug("step - info %s", info)

        self.logger.debug(
            "step return: obs: %s, rew: %6.3f, terminated: %s",
            observation,
            reward,
            terminated,
        )

        truncated = False

        if not GYM_COMPAT_MODE:
            # No support for episode truncation
            # But added to complete new gym step API
           
            return observation, float(reward), terminated, truncated, info

        else:
            
            if not GYM_NEW_STEP_API_ACTIVE:
                done = terminated or truncated
                return obs, reward, done, info
            else:
                return obs, reward, terminated, truncated, info

    
    def _get_obs_from_state(self, state: dict) -> dict:
        """Get observation from state dict.

        Args:
            state (dict): state dictionary

        Returns:
            dict: observation dictionary
        """
        return {key: state[key] for key in self.cfg.obs_keys}

    def _get_time_of_day(self, step: int) -> np.array:
        """Get the time of day given a the current step.

        Inspired by
        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/.

        Args:
            step (int): the current time step.

        Returns:
            np.array: array of shape (2,) that uniquely represents the time of day
                in circular fashion.
        """
        time_of_day = np.concatenate(  # pylint: disable=unexpected-keyword-arg
            (
                np.cos(2 * np.pi * step * self.cfg.time_step_len / 24),
                np.sin(2 * np.pi * step * self.cfg.time_step_len / 24),
            ),
            dtype=self.cfg.dtype,
        )
        return time_of_day


if __name__ == "__main__":

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = DiscreteActions(
        env
    )

    obs = wrapped_env.reset(seed=42)
    for _ in range(1000):
        action = wrapped_env.action_space.sample()
   #     print("type of action space: ")
   #     print(type(wrapped_env.action_space))
   #     print("type of action: ")
   #     print(type(action))
        obs, reward, terminated, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
