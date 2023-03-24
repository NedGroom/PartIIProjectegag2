

## The only changes in this wrapper is to add a few lines after getObsFromState, to transform the time of day into the TimeToDeparture.
## There will also need to be a field in the environment which is TimeOfDeparture. This will be set each episode in a later implementation, but here I will just assign it to a value.

import gym
from gym.spaces import Discrete, Tuple, Box
from typing import Optional, Union, Any
import random
import math
import numpy as np
import copy
import bauwerk
from bauwerk.envs.solar_battery_house import EnvConfig
from bauwerk.constants import (
    GYM_COMPAT_MODE,
    GYM_NEW_RESET_API_ACTIVE,
    GYM_RESET_INFO_DEFAULT,
    GYM_NEW_STEP_API_ACTIVE,
)



class wrapperParaAgent_newRewardNoHindsight(gym.Wrapper):

    pmax = 3700     # pmax = 1, for relative power action type
    GRID = 0        # action ID
    PV = 1          # action ID

    def __init__(self, env):

        super().__init__(env)
        print("finished old reset")
        self.time_step = np.array([])

        
        self.cfg.obs_keys.remove("time_of_day")
        self.cfg.obs_keys.append("time_until_departure")
        self.cfg.episode_len = 34;  # the car may return at 8am and leave at 5pm the next day = 33 hours
        self.cfg.grid_selling_allowed = False
        self.cfg.paper_max_charge_power = 3700
        self.cfg.paper_battery_capacity = 16000 #16kWH
        self.cfg.epsPV = 0.3
        self.cfg.epsSOC = 0.3
        self.total_rewards = 0
        
        print(self.load.num_steps)
  #      self.load.num_steps = self.cfg.episode_len *200
  #      self.solar.num_steps = self.cfg.episode_len *200
  #      self.solar.reset()
  #      self.load.reset()
  #      print(self.load.num_steps)

        
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
            "time_until_departure": gym.spaces.Discrete(self.cfg.episode_len + 1),
          #  "time_step_cont": gym.spaces.Box(
          #      low=0, high=self.cfg.episode_len + 1, shape=(1,), dtype=self.cfg.dtype )
            
            }
        obs_spaces = [obs_spaces[key] for key in self.cfg.obs_keys]         # Selecting the subset of obs spaces selected
        self.observation_space = gym.spaces.Tuple(obs_spaces)

        print("calling my reset from init")
        self.reset()
        print("finished my init")
        print(self.cfg.obs_keys)


    def step(self, action):

  #      print("stepping")
     #   action = np.float32(action) # Is this meant to be here?

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


        #action = float(action)  # getting the float value



        assert action < self.cfg.paper_max_charge_power + 1
        self.logger.debug("step - action: %1.3f", action)

        # Get old state
        load = self.state["load"]
        pv_generation = self.state["pv_gen"]
        cum_load = self.state["cum_load"]
        cum_pv_gen = self.state["cum_pv_gen"]
        cum_pv_used = self.state["cum_pv_used"]
        cum_EVmaxPV = self.state["cum_EVmaxPV"]
        cum_LmaxPV = self.state["cum_LmaxPV"]
        cum_cost = self.state["cum_cost"]
        



        print(action)
        #action = self.get_power_from_action(action)
        print(action)
        attempted_action = copy.copy(action)

        
        #charging_power = self.battery.charge(power=action)
        load_overflow_from_solar = max(0, load - pv_generation)
        solar_load_usage = min(pv_generation, load)
        solar_leftover = max(0, pv_generation - load)
        if grid_charge_action:
            realcharge_action = min(attempted_action, self.cfg.paper_battery_capacity - self.battery.b)
            solar_used = solar_load_usage
            net_load = realcharge_action + load_overflow_from_solar # EV charged on grid, load tries to do PV first
        else: # solar charge
            #these lines are if solar charge takes priority, but load does
           # realcharge_action = min(attempted_action, pv_generation) # EV charged on solar
           # solar_leftover = pv_generation - realcharge_action
           # solar_used = min(pv_generation, realcharge_action + load)
           # net_load = max(0, load - solar_leftover) # load is in grid, whichever couldnt fit in leftover solar
            solar_attempted_action = min(solar_leftover, attempted_action)
            realcharge_action = min(solar_attempted_action, self.cfg.paper_battery_capacity - self.battery.b)
            solar_used = solar_load_usage + realcharge_action
            net_load = load_overflow_from_solar

        # paper says max SoC is 1, we say it is capacity, so no dividing by capacity (as seen in paper)
        self.battery.b = self.battery.b + realcharge_action
        assert self.battery.b <= self.cfg.paper_battery_capacity
        assert not (net_load < 0)
        # Draw remaining net load from grid and get price paid
        #cost = self.grid.draw_power(power=net_load)
        cost = net_load * self.cfg.time_step_len * self.grid.base_price

        self.logger.debug("step - net load: %s", net_load)

        assert (pv_generation - solar_used  >= -0.000005)
        cum_load += load
        cum_pv_gen += pv_generation
        cum_pv_used += solar_used
        cum_EVmaxPV += solar_leftover
        cum_LmaxPV += solar_load_usage
        cum_cost += cost


        ## See if terminated, and calculate reward
        terminated = bool(self.time_til_departure == 1)
        if (terminated):
            reward = self.process_completed_episode()
        else:
            reward = 0
        assert self.time_til_departure >= 0


        # Add impossible control penalty to cost
        info["power_diff"] = np.abs(realcharge_action - float(attempted_action))
        if self.cfg.infeasible_control_penalty:
            reward -= info["power_diff"]
            self.logger.debug(
                "step - cost: %6.3f, power_diff: %6.3f", cost, info["power_diff"]
            )

        
        self.time_step += 1
        self.time_step = self.time_step%24
        assert (self.time_til_departure == self.state["time_until_departure"])
        self.time_til_departure -=1

        # Get load and PV generation for next time step
        new_load = self.load.get_next_load()
        load_change = load - new_load
        load = new_load

        new_pv_generation = self.solar.get_next_generation()
        pv_change = pv_generation - new_pv_generation
        pv_generation = new_pv_generation




        self.state = { 
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_generation], dtype=self.cfg.dtype),
            "battery_cont": np.array(self.battery.b, dtype=self.cfg.dtype),
            "time_step": int(self.time_step),
            "time_step_cont": self.time_step.astype(self.cfg.dtype),
            "cum_load": cum_load,
            "cum_pv_gen": cum_pv_gen,
            "cum_pv_used": cum_pv_used,
            "cum_EVmaxPV": cum_EVmaxPV,
            "cum_LmaxPV": cum_LmaxPV, 
            "cum_cost": cum_cost,
            "load_change": np.array([load_change], dtype=self.cfg.dtype),
            "pv_change": np.array([pv_change], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(step=self.time_step),
            "time_until_departure": self.time_til_departure
        }

   #     print("saved step timeofday")
   #     print("time " + str(self.state["time_of_day"]))
   #     print("ttd " + str(self.time_til_departure))
       # print(self.time_step)

        observation = self._get_obs_from_state(self.state)

        #terminated = bool(self.time_step >= self.cfg.episode_len)
        

        info["net_load"] = net_load
        info["charging_power"] = realcharge_action
        info["load"] = self.state["load"]
        info["pv_gen"] = self.state["pv_gen"]
        info["cost"] = cost
        info["battery_cont"] = self.battery.b
        info["time_step"] = int(self.time_step)

        info = {**info, **self.grid.get_info()}

        self.logger.debug("step - info %s", info)

        self.logger.debug(
            "step return: obs: %s, rew: %6.3f, terminated: %s",
            observation,
            reward,
            terminated,
        )

        # No support for episode truncation
        # But added to complete new gym step API
        truncated = False

 #       return observation, float(reward), terminated, truncated, info

        if not GYM_COMPAT_MODE:
            # No support for episode truncation
            # But added to complete new gym step API
            #print("not gym compat")
            return (observation, info["time_step"]), float(reward), terminated, truncated, info

        else:
            
            if not GYM_NEW_STEP_API_ACTIVE:
                done = terminated or truncated
                return (observation, info["time_step"]), float(reward), done, info
            else:
                print("else compatability")
                return observation, reward, terminated, truncated, info

    def process_completed_episode(self): 
        my_pv_consumption = 100 * self.state["cum_pv_used"] / self.state["cum_pv_gen"]
        cum_EVmaxPV = min(self.state["cum_EVmaxPV"], self.cfg.paper_battery_capacity - self.SoC_on_arrival)
        max_pv_consumption = 100 * (self.state["cum_LmaxPV"] + cum_EVmaxPV) / self.state["cum_pv_gen"]
        total_cost = self.state["cum_cost"]

        enough_pv_consumption = bool(max_pv_consumption - my_pv_consumption <= self.cfg.epsPV)
        enough_SOC = bool(1 - self.battery.b/self.cfg.paper_battery_capacity <= self.cfg.epsSOC)
        if enough_pv_consumption and enough_SOC:
            reward = 1
            self.total_rewards += 1
        else:
            reward = 0

        report = [ "Day: " + str(self.save_old_data_start//24) + ".." ,
                "Data left off: "+str(self.save_old_data_start) + ". Data new start: " + str(self.new_start) + "." ,
                "Car came home yesterday " + str(self.ep_start) + " , left " + str(self.ep_end) + " today, ep length: " + str(24-self.ep_start+self.ep_end),
                "Max consumption: " + str(max_pv_consumption) + ", PV consumption: " + str(my_pv_consumption) + " %. Total cost: " + str(total_cost),
                "Reward: " + str(reward)
            ]
        print(report)
        return reward

    def reset(        
        self,
        *,
        return_info: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict] = None,  # pylint: disable=unused-argument
    ) :
        """Resets environment to initial state and returns an initial observation.
        
        Returns:
            observation (object): the initial observation.
        """
    

        ## compatability changes so that super().reset() can run properly
        if hasattr(self,"_np_random"):
            nprandom = self._np_random
        else:
            nprandom=None
        time = self.time_step
        self.cfg.obs_keys.remove("time_until_departure")
        self.cfg.obs_keys.append("time_of_day")
        self.old_data_end = self.load.time_step
        assert self.solar.time_step == self.old_data_end

        initialobs = super().reset(seed=seed)

     #   self._np_random = nprandom
        self.time_step = time
        self.cfg.obs_keys.append("time_until_departure")#
        self.cfg.obs_keys.remove("time_of_day")


        

        #misc
        if self.force_task_setting and not self._task_is_set:
            raise RuntimeError(
                "No task set, but force_task_setting active. Have you set the task"
                " using `env.set_task(...)`?"
            )
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)


        ## CAR has just left the house in the morning
        
        if self.time_step.size == 0:
            self.time_step = np.array([17])
            time_of_arrival = random.randint(self.time_step + 1, 23)
            time_skipped = self.time_step   # so that the initial reset will take load data from 0 to 5pm
        else:
            assert self.time_step > np.array([6])
            assert self.time_step < np.array([20])  # should have called reset when car left the house between 7am-7pm
            time_of_arrival = random.randint(self.time_step + 1, 23)
            time_skipped = time_of_arrival - self.time_step
            self.time_step = np.array([time_of_arrival])
            
        self.ep_start = copy.copy(self.time_step)

        # update pv,load data to new return time
        self.save_old_data_start = copy.copy(self.old_data_end)
        self.new_start = (self.old_data_end + time_skipped)[0]
        
    #    self.load.fix_start(start=self.new_start)
    #    self.solar.fix_start(start=self.new_start)
        self.load.reset()
        self.solar.reset()
        self.load.time_step = self.new_start
        self.solar.time_step = self.new_start


        ## CAR has just arrived at home in the evening, sample battery content and travel plans

        self.SoC_on_arrival = random.uniform(0.2,0.5) * self.cfg.paper_battery_capacity  # according to paper
        self.battery.b = np.array([self.SoC_on_arrival], dtype=np.float32)

        time_of_departure = np.array([random.randint(7,19)]) #necessarily the next day
        self.ep_end = time_of_departure
        if self.time_step < 24:
            time_til_departure = 24 - self.time_step + time_of_departure
        else:
            time_til_departure = time_of_departure - self.time_step
        assert (time_til_departure > np.array([-1]))

        self.time_til_departure = time_til_departure
    #    print(time_til_departure)

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

    #    self.battery.reset()
    #    self.load.reset(start=start)
    #    self.solar.reset(start=start)

     #   print(self.load)

        load = self.load.get_next_load()
        pv_gen = self.solar.get_next_generation()

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_gen], dtype=self.cfg.dtype),
            "battery_cont": np.array(
                self.battery.get_energy_content(), dtype=self.cfg.dtype
            ),
            "time_step": self.time_step,
            "time_step_cont": np.array([self.time_step], dtype=self.cfg.dtype),
            "cum_load": np.array([0.0], dtype=self.cfg.dtype),
            "cum_pv_gen": np.array([0.0], dtype=self.cfg.dtype),
            "cum_pv_used": np.array([0.0], dtype=self.cfg.dtype),
            "cum_LmaxPV": np.array([0.0], dtype=self.cfg.dtype),
            "cum_EVmaxPV": np.array([0.0], dtype=self.cfg.dtype),
            "cum_cost": np.array([0.0], dtype=self.cfg.dtype),
            "load_change": np.array([0.0], dtype=self.cfg.dtype),
            "pv_change": np.array([0.0], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(self.time_step),
            "time_until_departure": time_til_departure
        }

        observation = self._get_obs_from_state(self.state)

        self.logger.debug("Environment reset.")

        if return_info:
            return_val = (observation, {})
        else:
            return_val = observation
    #    print("done new reset")
        return observation

    def get_power_from_action(self, action: object) -> object:
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if action > 0:
                action *= self.cfg.paper_max_charge_power
            else:
                action *= -self.cfg.paper_max_charge_power
            assert action >= 0
        return action

    def _get_obs_from_state(self, state: dict) -> dict:
        """Get observation from state dict.

        Args:
            state (dict): state dictionary

        Returns:
            dict: observation dictionary
        """
 #       print("Tring to call key in obs keys")
 #       print(self.cfg.obs_keys)
        return [state[key] for key in self.cfg.obs_keys]

    def _get_time_of_day(self, step: int) -> np.array:
        

        
        return step % 24
          
if __name__ == "__main__":

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = wrapperPartial_newRewardNoHindsight(    env    )
    print(wrapped_env.action_space)
    print(wrapped_env.observation_space)  # Discrete(21)

    obs = wrapped_env.reset(seed=42)
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, done, info = wrapped_env.step(action)
    #print(obs)

    for _ in range(15):
        while True:
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, done, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
            if (terminated):
                wrapped_env.reset()
                break
    print("Total rewards = " + str(wrapped_env.total_rewards))