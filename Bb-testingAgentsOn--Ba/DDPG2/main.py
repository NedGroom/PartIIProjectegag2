#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import sys
import os
import math
import bauwerk
from bauwerk.envs.solar_battery_house import EnvConfig, SolarBatteryHouseCoreEnv

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight
import matplotlib.pyplot as plt

#gym.undo_logger_setup()

def train(num_episodes, agent, env,  evaluate, validate_every, 
          output, num_sample_eps=30, debug=True, warmup=0, bsize=2, epsilon=2, seed=2, loadscaling=None):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.

    observation = deepcopy(env.reset(seed=seed))
    agent.reset(observation)
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
    
    while episode < num_episodes:

       # print(env.load.time_step)
        assert(env.time_step == env.load.time_step%24)

        # agent pick action ...
        if step <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env step, observe, and update policy
        observation, reward, done, _, info = env.step(action)
        agent.observe(reward, observation, done)
        if episode > warmup :
            agent.update_policy()

            if episode % validate_every == 0 and validate_every > 0 and countsampleepisodes < numsampleepisodes:
                eptimes = np.append(eptimes,info["time_step"])
                epindices = np.append(epindices, info["data_index"])
                eppvs = np.append(eppvs,info["pv_gen"])
                eploads = np.append(eploads,info["load"])
                epsocs = np.append(epsocs,info["battery_cont"])
                epcosts = np.append(epcosts,info["cum_cost"])
                epactions = np.append(epactions, info["realcharge_action"])

        # [optional] save intermideate model
     #   if episode % int(num_episodes/3) == 0:
     #       agent.save_model(output) ###was about to add a if correct episode store display data

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                episode_reward, True
            )
     #       print(info)
            pv_consums.append(info["my_pv_consumption"] / 100) # for percent
            maxpvs.append(info["max_pv_consumption"] / 100) # for percent
            socs.append(info["battery_cont"])
            rewards.append(episode_reward)
            totalcosts.append(info["total_cost"])



            # [optional] evaluate
            if episode > warmup and episode % validate_every == 0 and validate_every > 0:
                if countsampleepisodes < numsampleepisodes:
                    sampleepisodes.append((eptimes,epindices,eppvs,eploads,epsocs,epcosts,epactions))
                    countsampleepisodes +=1
                    eptimes = np.array([])
                    epindices = np.array([])
                    eppvs = np.array([])
                    eploads = np.array([])
                    epsocs = np.array([])
                    epcosts = np.array([])
                    epactions = np.array([])
                dataslice["pv_consums"] = pv_consums
                dataslice["maxpvs"] = maxpvs
                dataslice["socs"] = socs
                dataslice["rewards"] = rewards
                dataslice["costs"] = totalcosts
                dataslices.append(dataslice)
                
                policy = lambda x: agent.select_action(x, decay_epsilon=False)
                validate_reward, testrewards = evaluate(env, policy, stats=dataslice, debug=True, save=True)
                if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
                
                pv_consums, maxpvs, socs, rewards, totalcosts, dataslice = [], [], [], [], [], {}

            # reset
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            observation = env.reset(seed=seed)
            agent.reset(observation)

    namecfg = (output, bsize, epsilon, seed)
    path = '{}/bs{}eps{}seed{}'.format(*namecfg)
    if not os.path.exists(path):
        os.makedirs(path)
    plotsampleepisodeslong(sampleepisodes, path, loadscaling)
    save_results_with_data(testrewards, dataslices, namecfg, '{}/validate_slices_inside'.format(path), interval=validate_every)
 #   evaluate.save_results()

def save_results_with_data(testrewards, dataslices, namecfg, fn, interval=None):
        output, bsize, epsilon, seed = namecfg    


        yrew = np.mean(testrewards, axis=0)
        errorrew=np.std(testrewards, axis=0)

        yconsums = [np.mean(slice["pv_consums"]) for slice in dataslices]
        print(len(dataslices))
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

        x = range(interval,(testrewards.shape[1] + 1)*interval,interval)


        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.title('epsilon: {}, batch size: {}, seed: {}'.format(epsilon, bsize, seed))
        ax.errorbar(x, yrew, yerr=errorrew, fmt='-ko')
        ax.errorbar(x, yrewards, yerr=errorrewards, fmt='-ro')
        ax.errorbar(x, ycosts, yerr=errorcosts, fmt='-go')
        ax.errorbar(x, ysocs, yerr=errorsocs, fmt='-bo')
        ax.errorbar(x, yconsums, yerr=errorconsum, fmt='-co')
        ax.errorbar(x, ymaxpvs, yerr=errormaxpv, fmt='c.')
        
        ax.legend(['test av reward','interval av reward','interval av total costs', 'interval av SoCs','interval av pv consumption','interval av max pv consum'])

    #    ax.plot(x, ymaxpvs[None, :])
        plt.savefig(fn+'.png')
   #     savemat(fn+'.mat', {'reward':testrewards})
        plt.close()
        print("saved Average Reward")

def plotsampleepisodes(data, path):
    path = '{}/sample_episodes'.format(path)
    numeps = len(data)
    fig, axarr = plt.subplots(1, numeps)
    for ep in range(numeps):
    #    numsteps = range(len(data[ep][0]))
        timesteps, pvs, loads, socs, costs = data[ep]
        zeroat = np.where(timesteps == 0)[0][0]
        add = 24*np.append(np.zeros(zeroat), np.ones(len(timesteps)-zeroat))
        timesteps = timesteps + add

        axarr[ep].plot(timesteps, pvs)
        axarr[ep].plot(timesteps, loads)
        axarr[ep].plot(timesteps, socs)
        axarr[ep].plot(timesteps, costs / 1000)
        axarr[ep].legend(['eppvs','eploads','epsocs','epcosts'])
    plt.savefig(path+'.png')

def plotsampleepisodeslong(data, path, loadscaling):
    
    numeps = len(data)
    fig, axarr = plt.subplots(3, math.ceil(numeps/3))
    print(math.ceil(numeps/3))
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
        axarr[axida, axidb].plot(timesteps, pvs / loadscaling)    # this will call an error if train eps too small, or validateevery too big
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
        axarr2[axida, axidb].plot(indices, pvs / loadscaling)
        axarr2[axida, axidb].plot(indices, loads / loadscaling)
        axarr2[axida, axidb].plot(indices, socs)
        axarr2[axida, axidb].plot(indices, costs / 1000)
        axarr2[axida, axidb].plot(indices, realactions / 3500)
        
        axarr2[axida, axidb].axvspan(20, 31, alpha=0.25, color='grey')        
        if(8 in indices): axarr2[axida, axidb].axvspan(0, 7, alpha=0.25, color='grey')
        if(43 in indices): axarr2[axida, axidb].axvspan(44, 48, alpha=0.25, color='grey')


    axarr[0, 0].legend(['pv','load','soc','cost/1000','real actions'])
    axarr2[0, 0].legend(['pv','load','soc','cost/1000','real actions'])

    
    path = '{}/sample_episodes'.format(path)
    fig.savefig(path+'.png')
    figind.savefig(path+'indices.png')
    print("saved episodes")

def test(num_episodes, agent, env, evaluate, model_path, visualize=False, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward, testrewards = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


def main(mode='', train_eps=0, bsize=64, epsilon=50000, validate_eps=20, validate_every=1000, seed=1, warmup=0, saveload='default', loadscaling=None):

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='bauwerk/SolarBatteryHouse-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=500000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=500, type=int, help='train iters each timestep')
    parser.add_argument('--epsilondecay', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--saveload', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
   # args.output = get_output_folder(args.output, args.env)
    if saveload == 'default':
        args.saveload = 'output/{}-rundef'.format(args.env)
    else: 
        args.saveload = 'output/{}-{}'.format(args.env,saveload)
    args.mode = mode
    args.train_iter = train_eps + warmup
    args.bsize=bsize
    args.epsilon = epsilon
    args.validate_episodes = validate_eps
    args.validate_every = validate_every
    args.seed = seed
    args.warmup = warmup - 1
    num_sample_eps = 6

    
    #env = NormalizedEnv(gym.make(args.env))

    cfg = { 'solar_scaling_factor' : loadscaling,
          'load_scaling_factor' : loadscaling}
   # env = gym.make(args.env, cfg)
    env = SolarBatteryHouseCoreEnv(cfg)
    env = wrapperPartial_newRewardNoHindsight(env)
    print(env.cfg.solar_scaling_factor)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    num_states = len(env.observation_space)
    num_actions = env.action_space.shape[0]

    agent = DDPG(num_states, num_actions, args)
    evaluate = Evaluator(args.validate_episodes, interval=args.validate_every, save_path=args.saveload, args=args)


    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_every, args.saveload, 
              warmup=args.warmup, num_sample_eps = num_sample_eps, debug=args.debug, bsize=bsize, epsilon=epsilon, seed = args.seed, loadscaling=loadscaling)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.saveload,
             visualize=False, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


if __name__ == '__main__':
    main(mode='train')  