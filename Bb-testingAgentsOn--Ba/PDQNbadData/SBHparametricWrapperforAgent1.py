
import gym
from gym.spaces import Discrete, Tuple, Box
import math
from typing import Optional
import numpy as np
import bauwerk
import copy
from bauwerk.constants import (
    GYM_COMPAT_MODE,
    GYM_NEW_RESET_API_ACTIVE,
    GYM_RESET_INFO_DEFAULT,
    GYM_NEW_STEP_API_ACTIVE,
)


class ParametricActions(gym.ActionWrapper):

    pmax = 3700     # pmax = 1, for relative power action type
    GRID = 0        # action ID
    PV = 1          # action ID

    def __init__(self, env):
        super().__init__(env)

        parameters_min = np.array([0, 0], dtype="float32")                      # min of charge with grid and PV
        parameters_max = np.array([self.pmax, self.pmax], dtype="float32")      # max of charge with grid and PV
        self.action_space = Tuple((Discrete(2), Box(parameters_min[0:1], parameters_max[0:1]), Box(parameters_min[1:], parameters_max[1:])))
        obs_spaces = {
            "load": gym.spaces.Box(
                low=self.load.min_value,
                high=self.load.max_value,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "pv_gen": gym.spaces.Box(
                low=self.solar.min_value,
                high=self.solar.max_value,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "battery_cont": gym.spaces.Box(
                low=0, high=self.battery.size, shape=(1,), dtype=self.cfg.dtype
            ),
            "time_step": gym.spaces.Discrete(self.cfg.episode_len + 1),
            "time_step_cont": gym.spaces.Box(
                low=0, high=self.cfg.episode_len + 1, shape=(1,), dtype=self.cfg.dtype
            ),
            "cum_load": gym.spaces.Box(
                low=0, high=np.finfo(float).max, shape=(1,), dtype=self.cfg.dtype
            ),
            "cum_pv_gen": gym.spaces.Box(
                low=0, high=np.finfo(float).max, shape=(1,), dtype=self.cfg.dtype
            ),
            "load_change": gym.spaces.Box(
                low=np.finfo(float).min,
                high=np.finfo(float).max,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "pv_change": gym.spaces.Box(
                low=np.finfo(float).min,
                high=np.finfo(float).max,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "time_of_day": gym.spaces.Box(
                low=0, high=24.0, shape=(1,), dtype=self.cfg.dtype
            ),
        }
        # Selecting the subset of obs spaces selected
        obs_spaces = {key: obs_spaces[key] for key in self.cfg.obs_keys}
        self.observation_space = gym.spaces.Dict(obs_spaces)
        #self.reset()
        
    def action(self, act):
        choice, p1, p2 = act
        action = (choice, min(p1, self.pmax), min(p2, self.pmax))
        print("run")
        return action
    
    def step(self, action):         # this is the step from solar_battery_house.py copied, with only few lines changed. Will make clear where it is unchanged.
        #action = np.float32(action)
   #     print("type of action space: ")
   #     print(type(self.action_space))
   #     print("type of action: ")
   #     print(action)
   #     print(type(action))
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        info = {}

        id, parameters0, parameters1 = action
        if id == self.GRID:
            action = -parameters0
        elif id == self.PV:
            action = parameters1
        else: 
            raise ValueError('action not chosen from dicrete grid or pv') 

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
            print("not gym compat")
            return observation, float(reward), terminated, truncated, info

        else:
            
            if not GYM_NEW_STEP_API_ACTIVE:
                done = terminated or truncated
                return (observation, info["time_step"]), float(reward), done, info
            else:
                print("else compatability")
                return observation, reward, terminated, truncated, info


    def reset(
        self,
        *,
        return_info: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> object:
        """Resets environment to initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        if self.force_task_setting and not self._task_is_set:
            raise RuntimeError(
                "No task set, but force_task_setting active. Have you set the task"
                " using `env.set_task(...)`?"
            )
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        self.time_step = np.array([0])

        # make sure we have current type of battery
        # (relevant if task was changed)
        (
            self.min_charge_power,
            self.max_charge_power,
        ) = self.battery.get_charging_limits()

        if not self.cfg.data_start_index:
            start = np.random.randint((self.data_len // 24) - 1) * 24
        else:
            start = self.cfg.data_start_index

        self.battery.reset()
        self.load.reset(start=start)
        self.solar.reset(start=start)

        load = self.load.get_next_load()
        pv_gen = self.solar.get_next_generation()

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_gen], dtype=self.cfg.dtype),
            "battery_cont": np.array(
                self.battery.get_energy_content(), dtype=self.cfg.dtype
            ),
            "time_step": 0,
            "time_step_cont": np.array([0.0], dtype=self.cfg.dtype),
            "cum_load": np.array([0.0], dtype=self.cfg.dtype),
            "cum_pv_gen": np.array([0.0], dtype=self.cfg.dtype),
            "load_change": np.array([0.0], dtype=self.cfg.dtype),
            "pv_change": np.array([0.0], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(self.time_step),
        }

        observation = self._get_obs_from_state(self.state)

        self.logger.debug("Environment reset.")

    #    if return_info:
    #        return_val = (observation, {})
    #    else:
    #        return_val = observation

    #    return return_val

        return observation
    
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
        hour = time_of_day[0]
        mins = time_of_day[1]
        time_of_day = np.array([hour + mins],dtype=self.cfg.dtype)
   #     print("new time")
        return time_of_day

    def _get_obs_from_state(self, state: dict) -> dict:
        """Get observation from state dict.

        Args:
            state (dict): state dictionary

        Returns:
            dict: observation dictionary
        """
        return {key: state[key] for key in self.cfg.obs_keys}


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
#        print(action)
#        print("typee of action space: ")
#        print(type(wrapped_env.action_space))
        obs, reward, terminated, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
