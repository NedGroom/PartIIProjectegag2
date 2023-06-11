
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




def showPVandLoadDays(days:int, loadscale:int, newWrapper:bool, run:int):


    observation = None
    pvs = np.array([])
    loads = np.array([])

    cfg = { 'solar_scaling_factor' : loadscale,
          'load_scaling_factor' : loadscale,
          'infeasible_control_penalty': True}
    env = SolarBatteryHouseCoreEnv(cfg)
    
    if newWrapper:
        env = NewBaseEnvWrapper( env )
    else:
        env = wrapperPartial_newRewardNoHindsight( env )



    for step in range(days * 24):

        pvs = np.append(pvs, env.solar.get_next_generation())
        loads = np.append(loads, env.load.get_next_load())


    path = 'output/{}-{}'.format('bauwerk/exampleDayData','runTest'+str(run))
    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/loadscale{}solarscale{}seed{}newWrapper{}'.format(path, loadscale, loadscale, "hm", newWrapper)
    path = '{}showLoadsPVs'.format(path)


    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.xlabel('Timestep')
    plt.ylabel('grey is 8pm-7am')
    plt.title('No resetting, just examining data')
    x = range(1,len(pvs)+1)
    ax.plot(x, pvs, '-ko')
    ax.plot(x, loads, '-ro')
    for i in range(days):
        ax.axvspan(20+24*i, 31+24*i, alpha=0.25, color='grey')
    ax.legend(['pv gen','load'])

    plt.savefig(path+'.png')
    print("saved Average Reward")


def showPVandLoadEpisodes(eps:int, loadscale:int, seed:int, run:int, wrapped:bool, showactions:bool):   ## this is resetting the environment on episode which i dont want
                            ## maybe could be useful to mark where the resets are, but for now no

    observation = None
    pvs = np.array([])
    loads = np.array([])
    times = np.array([])
    indices = np.array([])
    realactions = np.array([])


    cfg = { 'solar_scaling_factor' : loadscale,
            'load_scaling_factor' : loadscale,
            'data_start_index': None,
          'infeasible_control_penalty': True }
    env = SolarBatteryHouseCoreEnv(cfg)
    
    if wrapped:
        env = NewBaseEnvWrapper( env , seed=seed)


    print(env.cfg.infeasible_control_penalty)

    adjust = 0
    time = 0

    for episode in range(eps):

        # reset at the start of episode
        observation = env.reset(seed=seed)
        assert observation is not None

        # start episode
        done = False

        if wrapped:
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = np.array([np.random.uniform(-1.,1.)], dtype=np.float32)
                observation, reward, done, _, info = env.step(action)
                pvs = np.append(pvs, info["pv_gen"][0])
                loads = np.append(loads, info["load"][0])
                if showactions: realactions = np.append(realactions, info["realcharge_action"])
                if time > info["time_step"] + adjust: # have just ended a day
                    adjust += 24
                time = info["time_step"] + adjust
                times = np.append(times, time)
        elif not wrapped:
            for i in range(100):
                # basic operation, action ,reward, blablabla ...
                action = np.array([np.random.uniform(-1.,1.)], dtype=np.float32)
                observation, reward, done, _, info = env.step(action)
                pvs = np.append(pvs, info["pv_gen"][0])
                loads = np.append(loads, info["load"][0])
                if showactions: realactions = np.append(realactions, info["realcharge_action"])
                if time > info["time_step"] + adjust: # have just ended a day
                    adjust += 24
                time = info["time_step"] + adjust
                times = np.append(times, time)
        episode += 1

    #      if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            
  #  print("adjusted this times: " + str(adjust / 24))
    path = 'output/{}-{}'.format('bauwerk/exampleEpData','runTest'+str(run))
    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/loadscale{}solarscale{}seed{}wrapped{}'.format(path, loadscale, loadscale, seed, wrapped)
    path = '{}showLoadsPVsResetting'.format(path)

  

    x = range(0,len(pvs))


    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    ax.title.set_text('Resetting at episode end, like in training')
    ax.plot(times, pvs, '-ko')
    ax.plot(times, loads, '-ro')
    if showactions: ax.plot(times, realactions, '-bo')
    for i in range(eps):
        ax.axvspan(20+24*i, 31+24*i, alpha=0.25, color='grey')
    ax.legend(['pv gen','load'])
    #ax.plot(x, pvs - loads, 'g')

#    ax.plot(x, ymaxpvs[None, :])
    plt.savefig(path+'.png')
    print("saved Average Reward")


   # print(pvs)
   # print(pvs[np.newaxis])
 #   TestData = np.append(times[np.newaxis], pvs[np.newaxis], axis=0)
    print(len(times))
    print(max(times))
    assert len(times) == len(pvs)
    print(len(pvs))
    TestData = { 'times': times, 'pvs': pvs, 'loads': loads} 

   # assert TestData.shape[0] == 2
  #  TestData = np.append(TestData, loads[np.newaxis], axis=0)
  #  assert TestData.shape[0] == 3
    np.save(path, TestData)
    return (path, TestData)
   # print(TestData)


def isDataSame(datas: None, path1, path2, path3):

    if datas is None:
        files = glob.glob(path1 + '*.npy')
        files += glob.glob(path2 + '*.npy')
        files += glob.glob(path3 + '*.npy')
      #  print(files)

        graphs = []

        for file in files:
            data = np.load(file)
            graphs.append(data)

        answer = np.array_equal(graphs[0], graphs[1]) and  np.array_equal(graphs[1], graphs[2])

    else:
        answer = np.array_equal(datas[0], datas[1]) and  np.array_equal(datas[1], datas[2])

    return answer



def splitDays(data):
    times = data['times']


    slicestarts = []
    sliceends = []
    for i, time in enumerate(times):
        if i == 0:
            slicestarts.append(i)
            continue
        if time-1 != times[i-1]:
            slicestarts.append(i)
            sliceends.append(i-1)
    sliceends.append(len(data)-1)
    assert len(sliceends) == len(slicestarts)

    slicedicts = []

    for i in range(len(sliceends)):
        slicedict = {}
        for (key, list) in data.items():
            slicedict[key] = list[slicestarts[i]:sliceends[i]+1]
        print("dict", slicedict)
        slicedicts.append( slicedict )

    return (slicedicts)


def collateSavedData(data:None, run:int):

    path = 'output/bauwerk/exampleEpData-runTest' + str(run) + '/'

    if data is None:
        files = glob.glob(path + '*.npy')
        graphs = []
        for file in files:
            data = np.load(file)
            graphs.append(data)
    else: 
        graphs = data


    numgraphs = len(graphs)
    fig, axarr = plt.subplots(3, math.ceil(numgraphs/3))
    figind, axarr2 = plt.subplots(3, math.ceil(numgraphs/3))

    fig.figsize = (40, 4*math.ceil(numgraphs/3))
    figind.figsize = (40, 4*math.ceil(numgraphs/3))

    mintimes, maxtimes = [], []

    for graph in range(numgraphs):

    #    print("new graph")
    #    print(graphs[graph])

        times = graphs[graph]['times']
        mintimes.append(min(times))
        maxtimes.append(max(times))
        dataslices = splitDays(graphs[graph])
     #   pvslices = (pvs,times)
     #   loadslices = splitDays(loads, times)
        
        
        axidb = math.floor(graph/3)
        axida = graph - 3 * axidb
        if axarr.shape == (3,):
            plotindex = graph
        else:
            plotindex = (axida, axidb)

        #zeroat = np.where(times == 0)[0]
        #print(zeroat)
        #add = 24*np.append(np.zeros(zeroat), np.ones(len(times)-zeroat))
        #print(add)
        #print(len(times))
        #times = times + add
        
        for slice in dataslices:    
            axarr[plotindex].plot(slice['times'], slice['pvs'], color='red')
            assert len(slice['times']) == len(slice['pvs'])
  #         print("ovs", slice['pvs'])
            axarr[plotindex].plot(slice['times'], slice['loads'], color='blue')
           
        

        axarr[plotindex].axvspan(20, 31, alpha=0.25, color='grey')
        print(times)
        if(8 > min(times)): axarr[plotindex].axvspan(0, 7, alpha=0.25, color='grey')
        if(43 < max(times)): axarr[plotindex].axvspan(44, 55, alpha=0.25, color='grey')
        if(68 < max( times)): axarr[plotindex].axvspan(68, 79, alpha=0.25, color='grey')
        if(92 < max( times)): axarr[plotindex].axvspan(92, 103, alpha=0.25, color='grey')


    axarr[plotindex].legend(['pv','load','soc','cost/1000','real actions'])

    mintime = min(mintimes)
    maxtime = max(maxtimes)
    for ax in axarr:
        ax.set_xlim([mintime,maxtime])

    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}collatedData'.format(path)
    fig.savefig(path+'.png')



#showPVandLoadDays(days=6, loadscale=3000, newWrapper=True)

def checkConsistentSeeds(wrapped:bool, run:int):
    path1, data1 = showPVandLoadEpisodes(eps=4, loadscale=3000, seed=1000, run=run, wrapped=wrapped, showactions = False)
    path2, data2 = showPVandLoadEpisodes(eps=4, loadscale=3000, seed=1000, run=run, wrapped=wrapped, showactions = False)
    path3, data3 = showPVandLoadEpisodes(eps=4, loadscale=3000, seed=1001, run=run, wrapped=wrapped, showactions = False)
    #print(isDataSame(path1, path2, path3))
    print(np.array_equal(data1,data2) and np.array_equal(data2,data3))
    data = [data1, data2, data3]
    collateSavedData(data, run=run)

checkConsistentSeeds(wrapped=True, run=10)

#showPVandLoadEpisodes(eps=4, loadscale=3001, newWrapper=True, seed=1000, run=4)
#showPVandLoadEpisodes(eps=4, loadscale=3000, newWrapper=True, seed=2000, run=4)
#showPVandLoadEpisodes(eps=4, loadscale=3001, newWrapper=True, seed=2000, run=4)
#showPVandLoadEpisodes(eps=4, loadscale=3000, newWrapper=True, seed=3000, run=4)
#showPVandLoadEpisodes(eps=4, loadscale=3001, newWrapper=True, seed=3000, run=4)


#collateSavedData(run=5)

