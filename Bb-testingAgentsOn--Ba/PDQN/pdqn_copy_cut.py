import os
import sys

import click
import argparse
import time
import matplotlib.pyplot as plt
import math
import gym

import bauwerk
from .wrapperParaAgent_newRewardNoHindsight import wrapperPara_newRewardNoHindsight
from bauwerk.envs.solar_battery_house import SolarBatteryHouseCoreEnv
from .parametricOverBase import ParametricOverBase
from newBaseEnvWrapper import NewBaseEnvWrapper

import numpy as np
from .agents.pdqn import PDQNAgent
from .common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper, PlatformFlattenedActionWrapper

import click
import ast


class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)




def run(seed:int = 4, episodes:int = 100, saveload='default',measure_step=30,loadscaling=5,num_sample_eps=4, evaluation_episodes:int = 30, batch_size:int = 32, gamma:float=0.99, inverting_gradients:bool=True, initial_memory_threshold:int = 500,
        use_ornstein_noise:bool=True, replay_memory_size:int = 50000, epsilon_steps:int = 1000, tau_actor:float=0.001, tau_actor_param:float=0.001, learning_rate_actor:float=0.0001,
        scaleState:bool=False, learning_rate_actor_param:float=0.001, epsilon_final:float=0.01, zero_index_gradients:bool=False, initialise_params:bool=True, scale_actions:bool=True,
        infeascontrol=False, distanceTargetReward=False, evalseed=123,clip_grad:float=10., indexed:bool=True, layers='[128,]', multipass:bool=False, weighted:bool=False, average:bool=False, random_weighted:bool=False, render_freq:int = 100,
        save_freq:int = 0, save_dir:str="results/platform", save_frames:bool=False, visualise:bool=False, action_input_layer:int = 0, title:str="PDDQN", tolerance=0.3):

   

    try:
        layers = ast.literal_eval(layers)
    except Exception as e:
        print(e)
        raise click.BadParameter(layers)
    

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (save_frames and visualise)
    
    env_name = 'bauwerk/SolarBatteryHouse-v0'
    cfg = { 'solar_scaling_factor' : loadscaling,
          'load_scaling_factor' : loadscaling,
          'infeasible_control_penalty': infeascontrol}
    env = SolarBatteryHouseCoreEnv(cfg)
    #env = wrapperPara_newRewardNoHindsight(env)
    env = NewBaseEnvWrapper(env, tolerance=tolerance, seed=seed, distanceTargetReward=distanceTargetReward)
    env = ParametricOverBase(env, scalingstate = scaleState)

  #  print("pre-scaling Observation space: ")
  #  print(env.observation_space)
  #  print("pre-scaling Action space: ")
  #  print(env.action_space)
   
    initial_params_ = [30., 3000.]

    if scaleState:
        env = ScaledStateWrapper(env)    # why do i need to scale the observation space
   # env = PlatformFlattenedActionWrapper(env)
#    if scale_actions:
#    env = ScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir,title)
  #  env.seed(seed)
    np.random.seed(seed)

    print("Observation space: ")
    print(env.observation_space)
    print("Action space: ")
    print(env.action_space)



    agent_class = PDQNAgent

    print("initialise agent")
    agent = agent_class(
                       env.observation_space, env.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       weighted=weighted,
                       average=average,
                       random_weighted=random_weighted,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed)
    print("agent instantiated")
    if initialise_params:
        print("action space.spaces")
        print(env.action_space.spaces)
        print("obs space.shape[0]")
        print(env.observation_space.shape[0])
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print("Agent: ")
    print(agent)
    max_steps = 128  * 36 - 1
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0

    performance=[]
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
    # agent.epsilon_final = 0.
    # agent.epsilon = 0.
    # agent.noise = None

    mastereps, mastersocs, masterconsums, masterrewards = [], [], [], []

    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))

        if scaleState:
            obs, _ = env.reset(seed=seed)
        else:
            obs = env.reset(seed=seed)

   #     state = [i[0] for i in obs]
        state = np.array(obs, dtype=np.float32, copy=False)
       # print(obs)
        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)


        episode_reward = 0.
        agent.start_episode()
        terminal=False
        for j in range(max_steps):
   #         print(action)
            ret = env.step(action)
   #         (next_state, steps), reward, terminal, truncated, info = ret
            next_state, reward, terminal, truncated, info = ret
            #next_state = list(next_state.values())
           # next_state = [i[0] for i in next_state]
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal)#, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            
            action = next_action
            state = next_state

            episode_reward += reward
            if visualise and i % render_freq == 0:
                env.render()

            if i % measure_step == 0 and countsampleepisodes < numsampleepisodes:
                eptimes = np.append(eptimes,info["time_step"])
                epindices = np.append(epindices, info["data_index"])
                eppvs = np.append(eppvs,info["pv_gen"])
                eploads = np.append(eploads,info["load"])
                epsocs = np.append(epsocs,info["battery_cont"])
                epcosts = np.append(epcosts,info["cum_cost"])
                epactions = np.append(epactions, info["realcharge_action"])

            if terminal:
                break
        agent.end_episode()


        returns.append(episode_reward)
        total_reward += episode_reward
      #  if i % 100 == 0: # 100 episodes
      #      print(str(i))
     #       print("Total average return: " + str(total_reward / (i + 1)))
    #        print("Average return over last 100: " + str(np.array(returns[-100:]).mean()))
   #         print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))

        ##datasampling
        pv_consums.append(info["my_pv_consumption"] ) # for percent
        maxpvs.append(info["max_pv_consumption"] ) # for percent
        socs.append(info["battery_cont"])
        rewards.append(reward)
        totalcosts.append(info["total_cost"] / 1000)

        mastereps.append(i)
        masterconsums.append(info["my_pv_consumption"] / info["max_pv_consumption"])
        mastersocs.append(info["battery_cont"])
        masterrewards.append(reward)

        if i % measure_step == 0:
            performance.append(evaluate(env, agent, evaluation_episodes, scaleState, evalseed))
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
                pv_consums, maxpvs, socs, rewards, dataslice, totalcosts = [], [], [], [], {}, []

    end_time = time.time()
    print("PDQN Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    returns = env.total_rewards 
 #   print("Ave. return =", (returns) / episodes )
    
    
    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        
        evaluation_returns = evaluate(env, agent, evaluation_episodes, scaleState, evalseed)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)



    if saveload == 'default':
        output = 'output/master/{}'.format(env_name)
    else: 
        output = 'output/master/{}'.format(saveload)
    namecfg = (output, learning_rate_actor, learning_rate_actor_param, tau_actor)
    path = '{}/lra{}lrp{}tau{}'.format(*namecfg)
    plotsampleepisodeslong(sampleepisodes, path, loadscaling)

    if not os.path.exists(path):
        os.makedirs(path)
    plotperformance(performance, dataslices, namecfg, path, interval=measure_step)

    assert len(mastereps) == len(masterconsums)
    masterData = {'episodes': mastereps, 'pvconsums': masterconsums, 'socs': mastersocs, 'rewards': masterrewards}

    return masterData



def plotperformance(performance, dataslices, namecfg, fn, interval=None):
    output, learning_rate_actor, learning_rate_actor_param, tau_actor = namecfg    

    print(performance)
    #yrew = np.mean(testrewards, axis=0)
    #errorrew=np.std(testrewards, axis=0)

    yrew = [np.mean(slice) for slice in performance]
    errorrew=[0.1 for slice in performance]

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

    x = range(0,len(performance)*interval,interval)

    xless = range(interval,len(performance)*interval,interval)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('lr actor: {}, lr actor param: {}, tau: {}'.format(learning_rate_actor, learning_rate_actor_param, tau_actor))
   # ax.errorbar(x, yrew, fmt='-ko')

    ax.errorbar(xless, yrewards, fmt='-ro')
    ax.errorbar(xless, ycosts,  fmt='-go')
    ax.errorbar(xless, ysocs, fmt='-bo')
    ax.errorbar(xless, yconsums, fmt='-co')
    ax.errorbar(xless, ymaxpvs,  fmt='c.')
        
    ax.legend(['interval av reward','interval av total costs', 'interval av SoCs','interval av pv consumption','interval av max pv consum'], loc='lower right')


    plt.savefig(fn+'.png')
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


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act] = max(min(act_param, np.array([1], dtype=np.float32)), np.array([0], dtype=np.float32))
    return (act, params[0], params[1])


def evaluate(env, agent, episodes=1000, scaleState=False, evalseed=123):
    epsf = agent.epsilon_final
    eps = agent.epsilon
    noise = agent.noise
    agent.epsilon_final = 0.
    agent.epsilon = 0.
    agent.noise = None

    returns = []
    timesteps = []
    for _ in range(episodes):
        if scaleState:
            state, _ = env.reset(seed=evalseed)
        else:
            state = env.reset(seed=evalseed)
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
       #     state = [i[0] for i in state]
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
    #        (state, _), reward, terminal,_, info = env.step(action)
            state, reward, terminal,_, info = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    agent.epsilon_final = epsf
    agent.epsilon = eps
    agent.noise = noise
    return np.array(returns)









if __name__ == '__main__':
    run(episodes=2000, saveload='runPdqna250',num_sample_eps=6, loadscaling=250, seed=1, measure_step=30)
    run(episodes=2000, saveload='runPdqna250',num_sample_eps=6, loadscaling=250, seed=2, measure_step=30)