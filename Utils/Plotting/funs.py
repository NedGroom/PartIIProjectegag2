
import sys
import os
import numpy as np
from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight
from bauwerk.envs.solar_battery_house import SolarBatteryHouseCoreEnv
from newBaseEnvWrapper import NewBaseEnvWrapper
import matplotlib.pyplot as plt
import math
import torch
import gym
import glob


def splitDays(data):
    times = data['times']


    slicestarts = []
    sliceends = []
    print(times)
   # assert False
    for i, time in enumerate(times):
        if i == 0:
            slicestarts.append(i)
            continue
        if time-1 != times[i-1]:
            slicestarts.append(i)
            sliceends.append(i-1)
    sliceends.append(len(times)-1)
    print("sliceends: ",sliceends)
    assert len(sliceends) == len(slicestarts)

    slicedicts = []

    for i in range(len(sliceends)):
        slicedict = {}
        for (key, list) in data.items():
            slicedict[key] = list[slicestarts[i]:sliceends[i]+1]
        print("adding diff episodes dict", slicedict)
        slicedicts.append( slicedict )

    return (slicedicts)




def plotperformance(performance, dataslices, namecfg, fn, interval=None):
    output, lr, update_step = namecfg    

    print(performance)
    #yrew = np.mean(testrewards, axis=0)
    #errorrew=np.std(testrewards, axis=0)

    yrew = [slice[1] for slice in performance]
    errorrew=[0.1 for slice in performance]

    yconsums = [np.mean(slice["pv_consums"]) for slice in dataslices]
    errorconsum=[np.std(slice["pv_consums"]) for slice in dataslices]

    ysocs = [np.mean(slice["socs"]) for slice in dataslices]
    errorsocs = [np.std(slice["socs"]) for slice in dataslices]

    ymaxpvs = [np.mean(slice["maxpvs"]) for slice in dataslices]
    errormaxpv = [np.std(slice["maxpvs"]) for slice in dataslices]
              
    yrewards = [np.mean(slice["rewards"]) for slice in dataslices]
    errorrewards = [np.std(slice["rewards"]) for slice in dataslices]    

    maxcost = max( [max(slice["costs"]) for slice in dataslices] + [1] )
    ycosts = [np.mean(slice["costs"]/maxcost) for slice in dataslices]
    errorcosts = [np.std(slice["costs"]/maxcost) for slice in dataslices]

   # x = range(0,len(performance)*interval,interval)
  #  for a in x:
   #     assert a in ep
    x = [item[0] for item in performance]
    xless = range(interval,len(performance)*interval,interval)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('lr: {}, update step: {}'.format(lr, update_step))
  #  ax.errorbar(x, yrew, fmt='-ko')


    ax.errorbar(xless, yrewards, fmt='-ro')
    ax.errorbar(xless, ycosts,  fmt='-go')
    ax.errorbar(xless, ysocs, fmt='-bo')
    ax.errorbar(xless, yconsums, fmt='-co')
    ax.errorbar(xless, ymaxpvs,  fmt='c.')
        
    ax.legend(['interval av reward','interval av total costs', 'interval av SoCs','interval av pv consumption','interval av max pv consum'], loc='lower right')

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
    print(numeps)
    for ep in range(numeps):

        timesteps, indices, pvs, loads, socs, costs, realactions = data[ep]
        
        axidb = math.floor(ep/3)
        axida = ep - 3 * axidb
        print((axarr.shape))
        print(axida)
        print(axidb)
        if axarr.shape == (3,):
            plotindex = ep
        else:
            plotindex = (axida, axidb)

        zeroat = np.where(timesteps == 0)[0][0]
        add = 24*np.append(np.zeros(zeroat), np.ones(len(timesteps)-zeroat))
        timesteps = timesteps + add
        axarr[plotindex].plot(timesteps, pvs / loadscaling)
        axarr[plotindex].plot(timesteps, loads / loadscaling)
        axarr[plotindex].plot(timesteps, socs)
        axarr[plotindex].plot(timesteps, costs / 1000)
        axarr[plotindex].plot(timesteps, realactions / 3500)
        axarr[plotindex].axvspan(20, 31, alpha=0.25, color='grey')

        indices = indices % 24
        if 0 in indices:
            zeroat = np.where(indices == 0)[0][0]
            add = 24*np.append(np.zeros(zeroat), np.ones(len(indices)-zeroat))
            indices = indices + add
        axarr2[plotindex].plot(indices, pvs / loadscaling)
        axarr2[plotindex].plot(indices, loads / loadscaling)
        axarr2[plotindex].plot(indices, socs)
        axarr2[plotindex].plot(indices, costs / 1000)
        axarr2[plotindex].plot(indices, realactions / 3500)
        
        axarr2[plotindex].axvspan(20, 31, alpha=0.25, color='grey')        
        if(8 in indices): axarr2[plotindex].axvspan(0, 7, alpha=0.25, color='grey')
        if(43 in indices): axarr2[plotindex].axvspan(44, 48, alpha=0.25, color='grey')


    axarr[plotindex].legend(['pv','load','soc','cost/1000','real actions'])
    axarr2[plotindex].legend(['pv','load','soc','cost/1000','real actions'])


    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}sample_episodes'.format(path)
    fig.savefig(path+'.png')
    figind.savefig(path+'indices.png')