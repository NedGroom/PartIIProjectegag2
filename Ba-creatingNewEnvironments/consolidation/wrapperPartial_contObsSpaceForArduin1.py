## The only changes in this wrapper is to add a few lines after getObsFromState, to transform the time of day into the TimeToDeparture.
## There will also need to be a field in the environment which is TimeOfDeparture. This will be set each episode in a later implementation, but here I will just assign it to a value.

import gym
from gym.spaces import Discrete, Tuple, Box
from typing import Optional, Tuple, Union, Any
import math
import numpy as np
import bauwerk
from bauwerk.envs.solar_battery_house import EnvConfig



class wrapperPartial_contObsSpace(gym.Wrapper):

    pmax = 3700     # pmax = 1, for relative power action type
    GRID = 0        # action ID
    PV = 1          # action ID

    def __init__(self, env):

        super().__init__(env)


        self.cfg.obs_keys.remove("time_of_day")
        self.cfg.obs_keys.append("time_until_departure")
        print(self.cfg.obs_keys)
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
            "time_until_departure": gym.spaces.Discrete(self.cfg.episode_len + 1),
          #  "time_step_cont": gym.spaces.Box(
          #      low=0, high=self.cfg.episode_len + 1, shape=(1,), dtype=self.cfg.dtype )
            
            }
        obs_spaces = [obs_spaces[key] for key in self.cfg.obs_keys]         # Selecting the subset of obs spaces selected
        self.observation_space = gym.spaces.Tuple(obs_spaces)

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
     #   print(type(self.action_space))
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

        if return_info:
            return_val = (observation, {})
        else:
            return_val = observation

        return return_val

    def _get_obs_from_state(self, state: dict) -> dict:
        """Get observation from state dict.

        Args:
            state (dict): state dictionary

        Returns:
            dict: observation dictionary
        """
        return [state[key] for keys in self.cfg.obs_keys]

          
if __name__ == "__main__":

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = wrapperPartial_contObsSpace(    env    )
    print(wrapped_env.observation_space)  # Discrete(21)

    obs = wrapped_env.reset(seed=42)
    print(obs)

    for _ in range(1000):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
