

## The only changes in this wrapper is to add a few lines after getObsFromState, to transform the time of day into the TimeToDeparture.
## There will also need to be a field in the environment which is TimeOfDeparture. This will be set each episode in a later implementation, but here I will just assign it to a value.

import gym
from gym.spaces import Discrete, Tuple, Box
from typing import Optional, Tuple, Union, Any
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import bauwerk
from bauwerk.envs.solar_battery_house import EnvConfig

#import AB


def plotperformance(dataslices, namecfg, fn, interval=None):
    output, bsize, epsilon, seed = namecfg    

    yconsums = [np.mean(slice["pv_consums"]) for slice in dataslices]
    errorconsum=[np.std(slice["pv_consums"]) for slice in dataslices]

    ysocs = [np.mean(slice["socs"]) for slice in dataslices]
    errorsocs = [np.std(slice["socs"]) for slice in dataslices]

    ymaxpvs = [np.mean(slice["maxpvs"]) for slice in dataslices]
    errormaxpv = [np.std(slice["maxpvs"]) for slice in dataslices]
              
    yrewards = [np.mean(slice["rewards"]) for slice in dataslices]
    errorrewards = [np.std(slice["rewards"]) for slice in dataslices]    

    maxcost = max( [max(slice["costs"]) for slice in dataslices] )
    ycosts = [np.mean(slice["costs"]/maxcost) for slice in dataslices]
    errorcosts = [np.std(slice["costs"]/maxcost) for slice in dataslices]

    x = range(0,len(dataslices)*interval,interval)
  #  for a in x:
   #     assert a in ep
    #x = [item[0] for item in performance]
    #xless = range(interval,len(performance)*interval,interval)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('epsilon: {}, batch size: {}, seed: {}'.format(epsilon, bsize, seed))
  #  ax.errorbar(x, yrew, fmt='-ko')
  #  ax.plot(x, yrew, '-ko')
  #  ax.errorbar(xless, yrewards, yerr=errorrewards, fmt='-ro')
  #  ax.errorbar(xless, ycosts, yerr=errorcosts, fmt='-go')
  #  ax.errorbar(xless, ysocs, yerr=errorsocs, fmt='-bo')
  #  ax.errorbar(xless, yconsums, yerr=errorconsum, fmt='-co')
 #   ax.errorbar(xless, ymaxpvs, yerr=errormaxpv, fmt='c.')

    ax.errorbar(x, yrewards, fmt='-ro')
    ax.errorbar(x, ycosts,  fmt='-go')
    ax.errorbar(x, ysocs, fmt='-bo')
    ax.errorbar(x, yconsums, fmt='-co')
    ax.errorbar(x, ymaxpvs,  fmt='c.')
        
    ax.legend(['interval av reward','interval av total costs', 'interval av SoCs','interval av pv consumption','interval av max pv consum'])

#    ax.plot(x, ymaxpvs[None, :])
    plt.savefig(fn+'.png')
#     savemat(fn+'.mat', {'reward':testrewards})
    plt.close()
    print("saved Average Reward")



def plotsampleepisodeslong(data, path, loadscaling):
    
    numeps = len(data)
    fig, axarr = plt.subplots(3, math.ceil(numeps/3))
    figind, axarr2 = plt.subplots(3, math.ceil(numeps/3))

    fig.figsize = (40, 4*math.ceil(numeps/3))
    figind.figsize = (40, 4*math.ceil(numeps/3))

    for ep in range(numeps):

        timesteps, indices, pvs, loads, socs, costs, realactions = data[ep]
        
        axidb = math.floor(ep/3)
        axida = ep - 3 * axidb

        zeroat = np.where(timesteps == 0)[0][0]
        add = 24*np.append(np.zeros(zeroat), np.ones(len(timesteps)-zeroat))
        timesteps = timesteps + add
        axarr[axida, axidb].plot(timesteps, pvs)
        axarr[axida, axidb].plot(timesteps, loads / loadscaling)
        axarr[axida, axidb].plot(timesteps, socs)
        axarr[axida, axidb].plot(timesteps, costs / 1000)
        axarr[axida, axidb].plot(timesteps, realactions / 3500)
        axarr[axida, axidb].axvspan(20, 31, alpha=0.25, color='grey')

        indices = indices % 24
        if 0 in indices:
            zeroat = np.where(indices == 0)[0][0]
            add = 24*np.append(np.zeros(zeroat), np.ones(len(indices)-zeroat))
            indices = indices + add
        axarr2[axida, axidb].plot(indices, pvs)
        axarr2[axida, axidb].plot(indices, loads / loadscaling)
        axarr2[axida, axidb].plot(indices, socs)
        axarr2[axida, axidb].plot(indices, costs / 1000)
        axarr2[axida, axidb].plot(indices, realactions / 3500)
        
        axarr2[axida, axidb].axvspan(20, 31, alpha=0.25, color='grey')        
        if(8 in indices): axarr2[axida, axidb].axvspan(0, 7, alpha=0.25, color='grey')
        if(43 in indices): axarr2[axida, axidb].axvspan(44, 48, alpha=0.25, color='grey')


    axarr[0, 0].legend(['pv','load','soc','cost/1000','real actions'])
    axarr2[0, 0].legend(['pv','load','soc','cost/1000','real actions'])


    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/sample_episodes'.format(path)
    fig.savefig(path+'.png')
    figind.savefig(path+'indices.png')




from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight



def main(episodes, saveload,num_sample_eps, loadscaling, seed, measure_step):

    env_name = "bauwerk/SolarBatteryHouse-v0"
    env = gym.make(env_name)
    wrapped_env = wrapperPartial_newRewardNoHindsight(    env    )
    print(wrapped_env.action_space)
    print(wrapped_env.observation_space)  # Discrete(21)

    obs = wrapped_env.reset(seed=seed)
    action = wrapped_env.action_space.sample()
    
    capacity = wrapped_env.cfg.paper_battery_capacity
    maxpower = wrapped_env.cfg.paper_max_charge_power
    tlag = 5 # tunes how shortsighted the algorithm is

    dataslice={}
    dataslices=[]
    pv_consums, maxpvs, socs, rewards, totalcosts = [], [], [], [], []
    eptimes = np.array([])
    epindices = np.array([])
    eppvs = np.array([])
    eploads = np.array([])
    epsocs =np.array([])
    epcosts = np.array([])
    epactions = np.array([])
    sampleepisodes = []
    countsampleepisodes = 0
    numsampleepisodes = num_sample_eps

    terminated=False

    for i in range(episodes):
        while True:
            (load,pv,SoC,ttd), reward, terminated, done, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
            
            if(SoC < wrapped_env.cfg.paper_battery_capacity):
                if (pv-load>0): # add 0.000005 * loadscaling?
                    action = np.array([(pv-load)/wrapped_env.cfg.paper_max_charge_power], dtype="float32")
                elif (maxpower*(ttd - tlag) < capacity-SoC):   ## == if we dont have enough time to wait for better moment to charge
                    action = np.array([-1], dtype = "float32")
                else:
                    action = np.array([0], dtype = "float32")
            else:
                action = np.array([0], dtype = "float32")

            if i % measure_step == 0 and countsampleepisodes < numsampleepisodes:
                eptimes = np.append(eptimes,info["time_step"])
                epindices = np.append(epindices, info["data_index"])
                eppvs = np.append(eppvs,info["pv_gen"])
                eploads = np.append(eploads,info["load"])
                epsocs = np.append(epsocs,info["battery_cont"])
                epcosts = np.append(epcosts,info["cum_cost"])
                epactions = np.append(epactions, info["realcharge_action"])

            if terminated:

                wrapped_env.reset()

                ##datasampling
                pv_consums.append(info["my_pv_consumption"] / 100) # for percent
                maxpvs.append(info["max_pv_consumption"] / 100) # for percent
                socs.append(info["battery_cont"])
                rewards.append(reward)
                totalcosts.append(info["total_cost"] / 1000)

                if i % measure_step == 0:
                    if countsampleepisodes < numsampleepisodes:
                        sampleepisodes.append((eptimes,epindices,eppvs,eploads,epsocs,epcosts,epactions))
                        countsampleepisodes +=1
                        eptimes,epindices,eppvs,eploads,epsocs,epcosts,epactions = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
                    if i != 0:
                        dataslice["pv_consums"] = pv_consums
                        dataslice["maxpvs"] = maxpvs
                        dataslice["socs"] = socs
                        dataslice["rewards"] = rewards
                        dataslice["costs"] = totalcosts
                        dataslices.append(dataslice)
                        pv_consums, maxpvs, socs, rewards, dataslice = [], [], [], [], {}

                break

    print("Total rewards = " + str(wrapped_env.total_rewards))

    if saveload == 'default':
        output = 'output/{}-rundef'.format(env_name)
    else: 
        output = 'output/{}-{}'.format(env_name,saveload)
    namecfg = (output, episodes, loadscaling, seed)
    path = '{}/eps{}loadscale{}seed{}'.format(*namecfg)
    plotsampleepisodeslong(sampleepisodes, path, loadscaling)
    plotperformance(dataslices, namecfg, '{}/validate_slices_inside'.format(path), interval=measure_step)



if __name__ == "__main__":
    
    main(episodes=2000, saveload='runRBCA',num_sample_eps=6, loadscaling=5, seed=1, measure_step=100)
    main(episodes=2000, saveload='runRBCA',num_sample_eps=6, loadscaling=5, seed=2, measure_step=100)