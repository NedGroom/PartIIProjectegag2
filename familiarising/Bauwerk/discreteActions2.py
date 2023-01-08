#this is code trying to extend the class but it doesnt make sense

# Setup and helper code

import bauwerk
from bauwerk import SolarBatteryHouseCoreEnv
import gym
import numpy as np
from stable_baselines3 import DQN

@dataclass
class DiscreteSolarBatteryHouseCoreEnv(SolarBatteryHouseCoreEnv):

    def __init__(
        self,
        cfg: Union[EnvConfig, dict] = None,
        force_task_setting=False,
    ) -> None:
            if cfg is None:
                cfg = EnvConfig()
            elif isinstance(cfg, dict):
                cfg = EnvConfig(**cfg)
            self.cfg: EnvConfig = cfg
            self.force_task_setting = force_task_setting
            self._task_is_set = False

            self._setup_components()

            self.data_len = min(len(self.load.data), len(self.solar.data))

            # Setting up action and observation space
            if self.cfg.action_space_type == "absolute":
                (
                    act_low,
                    act_high,
                ) = self.battery.get_charging_limits()
            elif self.cfg.action_space_type == "relative":
                act_low = -1
                act_high = 1
            else:
                raise ValueError(
                    (
                        f"cfg.action_space_type ({self.cfg.action_space_type} invalid)."
                        " Must be one of either 'relative' or 'absolute'."
                    )
                )

            self.action_space = gym.spaces.Discrete(
                n=21,
                start=-10,
            )
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
                    low=-1.0, high=1.0, shape=(2,), dtype=self.cfg.dtype
                ),
            }

            # Selecting the subset of obs spaces selected
            obs_spaces = {key: obs_spaces[key] for key in self.cfg.obs_keys}
            self.observation_space = gym.spaces.Dict(obs_spaces)

            self.logger = logger
            bauwerk.utils.logging.setup_log_print_options()
            self.logger.debug("Environment initialised.")

            self.state = None

            self.reset()


    def get_power_from_action(self, action: object) -> object:
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if action > 0:
                action *= self.max_charge_power
            else:
                action *= -self.min_charge_power

        return action

    def get_action_from_power(self, power: object) -> object:
        action = power
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if action > 0:
                action = floor(action / self.max_charge_power)
            else:
                action = -floor(action / self.max_charge_power)

        return action

	

    