

## The only changes in this wrapper is to add a few lines after getObsFromState, to transform the time of day into the TimeToDeparture.
## There will also need to be a field in the environment which is TimeOfDeparture. This will be set each episode in a later implementation, but here I will just assign it to a value.

import gym
from gym.spaces import Discrete, Tuple, Box, Dict
from typing import Optional, Union, Any
import random
import math
import numpy as np
import copy
import bauwerk
from bauwerk.envs.solar_battery_house import EnvConfig



class NewBaseEnvWrapper(gym.Wrapper):

    pmax = 3700     # pmax = 1, for relative power action type


    def __init__(self, env, tolerance=0.3):

        super().__init__(env)

        self.solar.time_step -= 1
        self.load.time_step -= 1

        
        self.time_step = np.array([])

        
        self.cfg.obs_keys.remove("time_of_day")
        self.cfg.obs_keys.append("time_until_departure")
        self.cfg.episode_len = 34;  # the car may return at 8am and leave at 5pm the next day = 33 hours
        self.cfg.grid_selling_allowed = False
        self.cfg.paper_max_charge_power = 3700
        self.cfg.paper_battery_capacity = 16000 #16kWH
        self.cfg.epsPV = tolerance
        self.cfg.epsSOC = tolerance
        self.total_rewards = 0
        self.episode = 0
        self.my_pv_consumption = 0
        self.max_pv_consumption = 0
        self.total_cost = 0
        


        statehigh = np.array(
            [
                self.load.max_value,
                self.solar.max_value,
                self.cfg.paper_battery_capacity,
                self.cfg.episode_len + 1,
         #       1,                                     ## these last two is for option 2 of how to deliver achieved_goal
        #        1
            ],
            dtype=np.float32,
        )
        statelow = np.array(
            [
                self.load.min_value,
                self.solar.min_value,
                0,
                -1,
       #         0,
        #        0
            ],
            dtype=np.float32,
        )
        goallow = np.array([0,0],dtype=np.float32) # goal is [pvconsumption, SoC], which are both between 0 and 1.
        goalhigh = np.array([1,1],dtype=np.float32)

        self.observation_space = Box(statelow, statehigh, dtype=self.cfg.dtype)                 ## this is the chosen option, option 3, where state type is unchanged.
     #   self.observation_space = Dict( { "state": Box(statelow, statehigh, dtype=self.cfg.dtype) ,             ## this is option one of how to deliver the goals along with the state
     #                                   "achievedGoal": Box(goallow, goalhigh, dtype=self.cfg.dtype) , 
     #                                   "desiredGoal": Box(goallow, goalhigh, dtype=self.cfg.dtype) })

        self.reset(seed=0)

        


    def step(self, action):


     #   action = np.float32(action) # Is this meant to be here?

        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        info = {}

        action = float(action)  # getting the float value
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
        


        grid_charge_action = bool(action < 0)

        action = self.get_power_from_action(action)
        attempted_action = copy.copy(action)

        
        load_overflow_from_solar = max(0, load - pv_generation)
        solar_load_usage = min(pv_generation, load)
        solar_leftover = max(0, pv_generation - load)
        if grid_charge_action:
            realcharge_action = min(attempted_action, self.cfg.paper_battery_capacity - self.battery.b)
            solar_used = solar_load_usage
            net_load = realcharge_action + load_overflow_from_solar # EV charged on grid, load tries to do PV first
        else: # solar charge
            solar_attempted_action = min(solar_leftover, attempted_action)
            realcharge_action = min(solar_attempted_action, self.cfg.paper_battery_capacity - self.battery.b)
            solar_used = solar_load_usage + realcharge_action
            net_load = load_overflow_from_solar

        # paper says max SoC is 1, we say it is capacity, so no dividing by capacity (as seen in paper)
        self.battery.b = self.battery.b + realcharge_action
        assert self.battery.b <= self.cfg.paper_battery_capacity
        assert not (net_load < 0)
        # Draw remaining net load from grid and get price paid
        cost = net_load * self.cfg.time_step_len * self.grid.base_price

        self.logger.debug("step - net load: %s", net_load)

        assert (pv_generation - solar_used  >= -0.000005 * self.cfg.solar_scaling_factor)
        cum_load += load
        cum_pv_gen += pv_generation
        cum_pv_used += solar_used
        cum_EVmaxPV = min(cum_EVmaxPV + solar_leftover, self.cfg.paper_battery_capacity - self.SoC_on_arrival) # total amount car could have been charged with solar
        cum_LmaxPV += solar_load_usage
        cum_cost += cost

        if cum_pv_gen: 
            self.my_pv_consumption = cum_pv_used / cum_pv_gen
            self.max_pv_consumption = (cum_LmaxPV + cum_EVmaxPV) / cum_pv_gen
        enough_pv_consumption = bool(self.max_pv_consumption - self.my_pv_consumption <= self.cfg.epsPV)
        enough_SOC = bool(1 - self.battery.b/self.cfg.paper_battery_capacity <= self.cfg.epsSOC)

        ## See if terminated, and calculate reward
        terminated = bool(self.time_til_departure == 1)
        if (terminated):
            self.episode += 1
            self.total_cost = cum_cost
            
            if enough_pv_consumption and enough_SOC:
                reward = 1
                self.total_rewards += 1
            else: 
                reward = 0
            report = [ "Day: " + str(self.episode),
                "Data left off: "+str(self.save_data_timestep_on_reset) + ". Data new start: " + str(self.new_start) + "." ,
                "Car came home yesterday " + str(self.ep_start) + " , left " + str(self.ep_end) + " today, ep length: " + str(24-self.ep_start+self.ep_end),
                "Max consumption: " + str(self.max_pv_consumption) + ", PV consumption: " + str(self.my_pv_consumption) + " %. Capacity: " + str(self.battery.b/self.cfg.paper_battery_capacity) + ". Total cost: " + str(self.total_cost),
                "Reward: " + str(reward)
            ]
            print(report)
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
            "battery_cont": np.array(self.battery.b, dtype=self.cfg.dtype),  # kept large for state
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


        observation = self._get_obs_from_state(self.state)
        

        info["net_load"] = net_load
        info["charging_power"] = realcharge_action
        info["load"] = self.state["load"]
        info["pv_gen"] = self.state["pv_gen"]
        info["cost"] = cost
        info["battery_cont"] = self.battery.b/self.cfg.paper_battery_capacity # made relative for info
        info["time_step"] = int(self.time_step)
        info["my_pv_consumption"] = self.my_pv_consumption
        info["max_pv_consumption"] = self.max_pv_consumption
        info["total_cost"] = self.total_cost
        info["cum_cost"] = self.state["cum_cost"]
        info["data_index"] = self.solar.time_step
        info["realcharge_action"] = realcharge_action
        info["achieved_goal"] = np.array([1 - self.battery.b/self.cfg.paper_battery_capacity, self.max_pv_consumption - self.my_pv_consumption], dtype=np.float32)      # new line
                    ## the goal is [0,0]. The episode is a success if you are within a tolerance of battery approaching max capacity, and pv consumption approching max pv consumption.

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

        return observation, float(reward), terminated, truncated, info


    def reset(        
        self,
        *,
        return_info: bool = False,
        seed: Optional[int] = None,
        options: Optional[dict] = None,  # pylint: disable=unused-argument
    ) :
        """Resets environment to initial state and returns an initial observation.
        
        Returns:
            observation (object): the initial observation.
        """


        ## compatability changes so that super().reset() can run properly

        time = self.time_step
        self.cfg.obs_keys.remove("time_until_departure")
        self.cfg.obs_keys.append("time_of_day")
        self.data_timestep_on_reset = self.load.time_step
        assert self.solar.time_step == self.data_timestep_on_reset



        initialobs = super().reset(seed=seed)


        self.time_step = time
        self.cfg.obs_keys.append("time_until_departure")
        self.cfg.obs_keys.remove("time_of_day")

        

        #misc
        if self.force_task_setting and not self._task_is_set:
            raise RuntimeError(
                "No task set, but force_task_setting active. Have you set the task"
                " using `env.set_task(...)`?"
            )


        ## CAR has just left the house in the morning, or this is resetting after a reset before a step
                                                    # this is known as reset before step sets time (of arrival) to 
        
        if self.time_step.size == 0 or self.time_step < np.array([7]) or self.time_step > np.array([19]): # this means cannot be a timeofdeparture
            self.time_step = np.array([0])     # so that time skipped is correct from data starting at 0
            self.data_timestep_on_reset -= self.data_timestep_on_reset % 24
            time_of_arrival = random.randint(np.array([20]) + 1, 23) # declare initial car left house at 8pm to show not reached this point by stepping
        else:
            assert self.time_step > np.array([6])
            assert self.time_step < np.array([20])  # should have called reset when car left the house between 7am-7pm
            time_of_arrival = random.randint(self.time_step + 1, 23)
        time_skipped = time_of_arrival - self.time_step
        self.time_step = np.array([time_of_arrival])
            
        self.ep_start = copy.copy(self.time_step)

        # update pv,load data to new return time
        self.save_data_timestep_on_reset = copy.copy(self.data_timestep_on_reset)
        self.new_start = (self.data_timestep_on_reset + time_skipped)[0]
        
    #    self.load.fix_start(start=self.new_start)
    #    self.solar.fix_start(start=self.new_start)
        self.load.reset()       ## since self.cfg.data_start_index=0, the load.reset() will always start from 0 
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
            assert False
            time_til_departure = time_of_departure - self.time_step
        assert (time_til_departure > np.array([-1]))

        self.time_til_departure = time_til_departure



        load = self.load.get_next_load()
        pv_gen = self.solar.get_next_generation()
        self.solar.time_step -= 1
        self.load.time_step -= 1

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

        return return_val

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

        return np.array([float(state[key]) for key in self.cfg.obs_keys], dtype=np.float64)

    def _get_time_of_day(self, step: int) -> np.array:
        

        
        return step % 24
          
if __name__ == "__main__":

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    wrapped_env = wrapperPartial_newRewardNoHindsight(    env    )
    print("Action, Observation space")
    print(wrapped_env.action_space)
    print(wrapped_env.observation_space)  # Discrete(21)

    obs = wrapped_env.reset(seed=42)
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, done, info = wrapped_env.step(action)


    for _ in range(1):
        while True:
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, done, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
            if (terminated):
                wrapped_env.reset()
                break
    print("Total rewards = " + str(wrapped_env.total_rewards))