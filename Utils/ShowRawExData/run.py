
import sys
import os
import numpy as np
from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight
import matplotlib.pyplot as plt
import torch
import gym




def showPVandLoad(days=5):


    observation = None
    pvs = np.array([])
    loads = np.array([])

    env = wrapperPartial_newRewardNoHindsight( gym.make("bauwerk/SolarBatteryHouse-v0") )

    for step in range(days * 24):

        pvs = np.append(pvs, env.solar.get_next_generation())
        loads = np.append(loads, env.load.get_next_load())


    path = 'output/{}-{}'.format('bauwerk/SolarBatteryHouse-v0','runa')
    path = '{}/bs{}eps{}seed{}'.format(path, 10, 5000, 1234)
    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/showLoadsPVs'.format(path)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
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


def showPVandLoadResetting(eps=5):   ## this is resetting the environment on episode which i dont want
                            ## maybe could be useful to mark where the resets are, but for now no

    observation = None
    pvs = np.array([])
    loads = np.array([])
    times = np.array([])
    indices = np.array([])
    realactions = np.array([])


    env = wrapperPartial_newRewardNoHindsight( gym.make("bauwerk/SolarBatteryHouse-v0") )

    adjust = 0
    time = 0

    for episode in range(eps):

        # reset at the start of episode
        observation = env.reset()
        assert observation is not None

        # start episode
        done = False
        while not done:
            # basic operation, action ,reward, blablabla ...
            action = [np.random.uniform(-1.,1.)]
            observation, reward, done, _, info = env.step(action)
            pvs = np.append(pvs, info["pv_gen"][0])
            loads = np.append(loads, info["load"][0] / 5)
            indices = np.append(indices, info["data_index"])
            realactions = np.append(realactions, info["realcharge_action"] / 3500)
            if time > info["time_step"] + adjust: # have just ended a day
                adjust += 24
            time = info["time_step"] + adjust
            times = np.append(times, time)
        episode += 1

    #      if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            
    print("adjusted this times: " + str(adjust / 24))
    path = 'output/{}-{}'.format('bauwerk/SolarBatteryHouse-v0','runa')
    path = '{}/bs{}eps{}seed{}'.format(path, 10, 5000, 1234)
    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/showLoadsPVsResetting'.format(path)

  

    x = range(0,len(pvs))


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    ax[0].title.set_text('Resetting at episode end, like in training')
    ax[0].plot(times, pvs, '-ko')
    ax[0].plot(times, loads, '-ro')
    ax[0].plot(times, realactions, '-bo')
    for i in range(eps):
        ax[0].axvspan(20+24*i, 31+24*i, alpha=0.25, color='grey')
    ax[0].legend(['pv gen','load'])
    #ax.plot(x, pvs - loads, 'g')

    ax[1].title.set_text('Using indices, not timesteps')
    ax[1].plot(indices, pvs, '-ko')
    ax[1].plot(indices, loads, '-ro')
    ax[1].plot(indices, realactions, '-bo')

    for i in range(eps):
        ax[1].axvspan(20+24*i, 31+24*i, alpha=0.25, color='grey')


#    ax.plot(x, ymaxpvs[None, :])
    plt.savefig(path+'.png')
    print("saved Average Reward")



showPVandLoad(days=6)
showPVandLoadResetting(eps=10)