

## The only changes in this wrapper is to add a few lines after getObsFromState, to transform the time of day into the TimeToDeparture.
## There will also need to be a field in the environment which is TimeOfDeparture. This will be set each episode in a later implementation, but here I will just assign it to a value.

import gym
#from gym.spaces import Discrete, Tuple, Box
from typing import Optional, Union, Any
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import bauwerk
from bauwerk.envs.solar_battery_house import SolarBatteryHouseCoreEnv


      

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

    maxcost = 0
    for _,_,_,_,_,costs,_ in data:
        maxcost = max(maxcost, np.max(costs))

    for ep in range(numeps):

        timesteps, indices, pvs, loads, socs, costs, realactions = data[ep]
        
        axidb = math.floor(ep/3)
        axida = ep - 3 * axidb

        zeroat = np.where(timesteps == 0)[0][0]
        add = 24*np.append(np.zeros(zeroat), np.ones(len(timesteps)-zeroat))
        timesteps = timesteps + add
        axarr[axida, axidb].plot(timesteps, pvs / loadscaling)
        axarr[axida, axidb].plot(timesteps, loads / loadscaling)
        axarr[axida, axidb].plot(timesteps, socs)
        axarr[axida, axidb].plot(timesteps, costs / maxcost)
        axarr[axida, axidb].plot(timesteps, realactions / 3500)
        axarr[axida, axidb].axvspan(20, 31, alpha=0.25, color='grey')

        indices = indices % 24
        if 0 in indices:
            zeroat = np.where(indices == 0)[0][0]
            add = 24*np.append(np.zeros(zeroat), np.ones(len(indices)-zeroat))
            indices = indices + add
        axarr2[axida, axidb].plot(indices, pvs / loadscaling)
        axarr2[axida, axidb].plot(indices, loads / loadscaling)
        axarr2[axida, axidb].plot(indices, socs)
        axarr2[axida, axidb].plot(indices, costs / maxcost)
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



def main(episodes, saveload,num_sample_eps, loadscaling, seed, measure_step, randomType):

    env_name = "bauwerk/SolarBatteryHouse-v0"
    #env = gym.make(env_name, cfg)
    cfg = { 'solar_scaling_factor' : loadscaling,
          'load_scaling_factor' : loadscaling}
    env = SolarBatteryHouseCoreEnv(cfg)
    wrapped_env = wrapperPartial_newRewardNoHindsight(    env    )

    print("action then obs space")
    print(wrapped_env.action_space)
    print(wrapped_env.observation_space)  # Discrete(21) # what??

    obs = wrapped_env.reset(seed=seed)
    action = wrapped_env.action_space.sample()
    print(type(action))


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

    capacity = wrapped_env.cfg.paper_battery_capacity
    maxpow = wrapped_env.cfg.paper_max_charge_power

    if randomType == 'simpleRandom':
        simpleRandom, paraRandom = True, False
    elif randomType == 'paraRandom':
        simpleRandom, paraRandom = False, True

    for i in range(episodes):
        #print("new episode")
        while True:

            (load,pv,SoC,ttd), reward, terminated, done, info = wrapped_env.step(action)  # include truncated output if in Gym0.26
            
            if paraRandom:
                if(SoC < capacity): # divide by capacity as coming from state, not info
                    rnd = np.random.random()
                    if (pv-load > 0): #-0.000005 * loadscaling): 
                        if rnd > 0.5: # solar
                         #   action = np.array([2 * (rnd - 0.5) ],dtype="float32") #  random action
                            action = np.array([ 2 * (rnd - 0.5) * (pv-load)/maxpow ],dtype="float32") #  random sample from leftover relative power
                        else:         # grid
                            action = np.array([ - 2 * rnd ], dtype = "float32")# random sample relative maxpower
                    else:   # if no solar possible
                        action = np.array([ - rnd ], dtype = "float32")# random sample relative maxpower
                else: # no charge possible
                    action = np.array([0], dtype = "float32")
            elif simpleRandom:
                action = wrapped_env.action_space.sample()



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
    
    main(randomType='paraRandom' ,episodes=2000, saveload='runRandParaA',num_sample_eps=6, loadscaling=3000, seed=1, measure_step=100)
    main(randomType='paraRandom' ,episodes=2000, saveload='runRandParaB',num_sample_eps=6, loadscaling=3000, seed=1, measure_step=100)
    #main(randomType='simpleRandom' ,episodes=2000, saveload='runRandSimA',num_sample_eps=6, loadscaling=3000, seed=1, measure_step=100)
    #main(randomType='simpleRandom' ,episodes=2000, saveload='runRandSimB',num_sample_eps=6, loadscaling=3000, seed=1, measure_step=100)