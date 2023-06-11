

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
from funs import splitDays


def showPVandLoadEpisodes(eps:int, loadscale:int, seed:int, run:int, wrapped:bool, skipsmalleps=-1):   ## this is resetting the environment on episode which i dont want
                            ## maybe could be useful to mark where the resets are, but for now no

    observation = None
    pvs = np.array([])
    loads = np.array([])
    times = np.array([])
    indices = np.array([])


    cfg = { 'solar_scaling_factor' : loadscale,
            'load_scaling_factor' : loadscale,
            'data_start_index': None ,
            'load_data': None,
            'infeasible_control_penalty': True}

    env = SolarBatteryHouseCoreEnv(cfg)
    
    if wrapped:
        env = NewBaseEnvWrapper( env , seed=seed)

    adjust = 0
    time = 0
    episode=0

    SOCarr = []
    epstartsteps = []

    steps = 0

    while episode < eps:

        # reset at the start of episode
        observation = env.reset(seed=seed)
        assert observation is not None

        SOCarr.append(observation[2])
        epstartsteps.append(steps)

        # start episode
        done = False
        

        if wrapped:
            while not done:
                # basic operation, action ,reward, blablabla ...
                observation, reward, done, _, info = env.step(env.action_space.sample())
                pvs = np.append(pvs, info["pv_gen"][0])
                loads = np.append(loads, info["load"][0])
                if time > info["time_step"] + adjust: # have just ended a day
                    adjust += 24
                time = info["time_step"] + adjust
                times = np.append(times, time)
                steps +=1
        elif not wrapped:
            for i in range(100):
                # basic operation, action ,reward, blablabla ...
                observation, reward, done, _, info = env.step(env.action_space.sample())
                pvs = np.append(pvs, info["pv_gen"][0])
                loads = np.append(loads, info["load"][0])
                if time > info["time_step"] + adjust: # have just ended a day
                    adjust += 24
                time = info["time_step"] + adjust
                times = np.append(times, time)
                steps +=1

        if steps < skipsmalleps:
            print("skipped too small")
            pvs = np.array([])
            loads = np.array([])
            times = np.array([])
            indices = np.array([])
            episode -= 1
            SOCarr.pop()
        episode += 1



    path = 'Utils/TestData/{}-{}'.format('exampleTestData','runTest'+str(run))
    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/loadscale{}solarscale{}seed{}wrapped{}eps{}'.format(path, loadscale, loadscale, seed, wrapped, eps)

    x = range(0,len(pvs))

 #   print("timesteps: ", steps)
 #   dataslices = splitDays({'times':times, 'pvs': pvs, 'loads': loads})
 #   print(dataslices)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    ax.title.set_text('Resetting at episode end, like in training')

    dataslices = splitDays({'times':times, 'pvs': pvs, 'loads': loads})
    for slice in dataslices:    
        ax.plot(slice['times'], slice['pvs'], '-ko')
        ax.plot(slice['times'], slice['loads'], '-ro')


    for i in range(eps):
        ax.axvspan(20+24*i, 31+24*i, alpha=0.25, color='grey')
    ax.legend(['pv gen','load'])

    plt.savefig(path+'.png')
    print("saved Average Reward")

    print(len(times))
    print(max(times))
    assert len(times) == len(pvs)
    print(len(pvs))

    TestData = { 'times': times, 'pvs': pvs, 'loads': loads, 'SOCarr': SOCarr, 'EpStartSteps': epstartsteps} 

    print("len of epstartsteps: ", len(epstartsteps)) 
    print("len of SOCarr: ", len(SOCarr)) 

    np.savez(path, times=times, pvs=pvs, loads=loads, SOCarr=SOCarr, epstartsteps=epstartsteps)
    return (path, TestData)
   # print(TestData)

path1, data1 = showPVandLoadEpisodes(eps=1, loadscale=3000, seed=1001, run=4, wrapped=True, skipsmalleps=-10)
print(path1)