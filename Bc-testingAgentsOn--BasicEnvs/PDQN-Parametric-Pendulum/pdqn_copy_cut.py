import os
import sys


import click
import argparse
import time
import matplotlib.pyplot as plt
import math
import gym
from gym.spaces import Tuple, Box, Discrete


import numpy as np
from agents.pdqn import PDQNAgent
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

import click
import ast


class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)

def toPendulumActions( act, act_param, actionspace: Tuple ):  # return continuous actions
    
    if act == 0:
        return -act_param
    elif act ==1:
        return act_param
    else:
        assert False
        return (2, [])
        

def pad_action(act, act_param): #, pmax):     # not being used
    zero=np.array([0],dtype=np.float32)
    two = zero + 2
    truncated = min(two, max(zero,act_param))
    
    if act == 0:
        return (0, truncated , zero)
    elif act ==1:
        return (1, zero, truncated )
    else:
        assert False
        return (2, [])


def parametricActionSpace( actionspace: Box):
    newActionSpace = Tuple(( Discrete(2), Box(0., -actionspace.low, (1,), np.float32), Box(0., actionspace.high, (1,), np.float32) ))
    return newActionSpace

def run(seed:int = 4, episodes:int = 100, saveload='default',measure_step=30,loadscaling=5,num_sample_eps=4, evaluation_episodes:int = 30, batch_size:int = 32, gamma:float=0.99, inverting_gradients:bool=True, initial_memory_threshold:int = 500,
        use_ornstein_noise:bool=True, replay_memory_size:int = 50000, epsilon_steps:int = 1000, tau_actor:float=0.01, tau_actor_param:float=0.001, learning_rate_actor:float=0.0001,
        learning_rate_actor_param:float=0.001, epsilon_final:float=0.01, zero_index_gradients:bool=False, initialise_params:bool=True, scale_actions:bool=True,
        clip_grad:float=10., indexed:bool=True, layers='[128,]', multipass:bool=False, weighted:bool=False, average:bool=False, random_weighted:bool=False, render_freq:int = 100,
        save_freq:int = 0, save_dir:str="results/platform", save_frames:bool=False, visualise:bool=False, action_input_layer:int = 0, title:str="PDDQN"):

   

    try:
        layers = ast.literal_eval(layers)
    except Exception as e:
        print(e)
        raise click.BadParameter(layers)
    

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (save_frames and visualise)
    dir = os.path.join(save_dir,title)
    np.random.seed(seed)
    
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    print("Observation space: ")
    print(env.observation_space)
    print("Action space: ")
    print(env.action_space)
    initial_params_ = [2, 2]

    ParametricPendulumActionSpace = parametricActionSpace(env.action_space)
    print("Para pendulum action space: ")
    print(ParametricPendulumActionSpace)

    # env = ScaledStateWrapper(env)     why do i need to scale the observation space
#    if scale_actions:
 #       env = ScaledParameterisedActionWrapper(env)


    assert not (multipass)
    agent_class = PDQNAgent
    if multipass:
        print("miltipass")
        assert 0
        agent_class = MultiPassPDQNAgent
    print("initialise agent")
    agent = agent_class(
                       env.observation_space, ParametricPendulumActionSpace,
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
        print("initialising params")
        print("info sliding_action_space.spaces[0].n for horizontal shape of initial weights, and len of initial bias, should be same as initial params len")
        print(ParametricPendulumActionSpace.spaces[0].n)
        print("info obs space.shape[0] for horizontal shape of initial weights") 
        print(env.observation_space.shape[0])
        initial_weights = np.zeros((ParametricPendulumActionSpace.spaces[0].n, env.observation_space.shape[0]))
        initial_bias = np.zeros(ParametricPendulumActionSpace.spaces[0].n)
        for a in range(ParametricPendulumActionSpace.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print("Agent: ")
    print(agent)
    max_steps = 200
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0

    performance=[]
    dataslice={}
    dataslices=[]
    pv_consums, maxpvs, socs, rewards, totalcosts = [], [], [], [], []
    eptimes = np.array([])
    eprewards = np.array([])
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

    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))

        obs = env.reset()
        state = obs[0]
     #   state = np.array(state, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent.act(state)
    #    action = pad_action(act, act_param) #, env.pmax)
        action = toPendulumActions(act, act_param, ParametricPendulumActionSpace)

        episode_reward = 0.
        agent.start_episode()
        terminal=False
        for j in range(max_steps):

            ret = env.step(action)
            
            next_state, reward, terminal,_, info = ret

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
         #   next_action = pad_action(next_act, next_act_param) #, env.pmax)
            next_action = toPendulumActions(next_act, next_act_param, ParametricPendulumActionSpace)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, j)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            
            action = next_action
            state = next_state

            episode_reward += reward

            if visualise and i % render_freq == 0:
                env.render()

            if i % measure_step == 0 and countsampleepisodes < numsampleepisodes:
                eptimes = np.append(eptimes,j)
                eprewards = np.append(eprewards, reward)


            if terminal:
                print("broken")
                break

        agent.end_episode()


        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0: # 100 episodes
            print(str(i))
            print("Total average return: " + str(total_reward / (i + 1)))
            print("Average return over last 100: " + str(np.array(returns[-100:]).mean()))
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))

        ##datasampling
        rewards.append(episode_reward)

        if i % measure_step == 0:
            performance.append(evaluate(env, agent, evaluation_episodes, ParametricPendulumActionSpace))
            if countsampleepisodes < numsampleepisodes:
                sampleepisodes.append((eptimes,eprewards))  #epindices,eppvs,eploads,epsocs,epcosts,epactions))
                countsampleepisodes +=1
                eptimes,eprewards = np.array([]),np.array([])
            if i != 0:
                dataslice["rewards"] = rewards
                dataslices.append(dataslice)
                pv_consums, maxpvs, socs, rewards, dataslice = [], [], [], [], {}

    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    print("Ave. return =", sum(returns) / len(returns))
    

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        
        evaluation_returns = evaluate(env, agent, evaluation_episodes, ParametricPendulumActionSpace)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)



    if saveload == 'default':
        output = 'output/{}-rundef'.format(env_name)
    else: 
        output = 'output/{}-{}'.format(env_name,saveload)
    namecfg = (output, batch_size, learning_rate_actor_param, seed)
    path = '{}/bs{}lr{}seed{}'.format(*namecfg)
    plotsampleepisodeslong(sampleepisodes, path, loadscaling)
    plotperformance(performance, dataslices, namecfg, '{}/validate_slices_inside'.format(path), interval=measure_step)



def plotperformance(performance, dataslices, namecfg, fn, interval=None):
    output, bsize, epsilon, seed = namecfg    

    print(performance)

    yrew = [np.mean(slice) for slice in performance]
    errorrew=[0.1 for slice in performance]
              
    yrewards = [np.mean(slice["rewards"]) for slice in dataslices]
    errorrewards = [np.std(slice["rewards"]) for slice in dataslices]    


    x = range(0,len(performance)*interval,interval)
    xless = range(interval,len(performance)*interval,interval)


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('epsilon: {}, batch size: {}, seed: {}'.format(epsilon, bsize, seed))

    ax.errorbar(x, yrew, fmt='-ko')
    ax.errorbar(xless, yrewards, fmt='-ro')

    ax.legend(['test av reward','interval av reward','interval av total costs', 'interval av SoCs','interval av pv consumption','interval av max pv consum'])


    plt.savefig(fn+'.png')
    plt.close()
    print("saved Average Reward")



def plotsampleepisodeslong(data, path, loadscaling):
    
    numeps = len(data)
    fig, axarr = plt.subplots(3, math.ceil(numeps/3))

    fig.figsize = (40, 4*math.ceil(numeps/3))

    for ep in range(numeps):

        timesteps, rewards  = data[ep]
        
        axidb = math.floor(ep/3)
        axida = ep - 3 * axidb
        if axarr.shape == (3,):
            plotindex = ep
        else:
            plotindex = (axida, axidb)

        zeroat = np.where(timesteps == 0)[0][0]
        add = 24*np.append(np.zeros(zeroat), np.ones(len(timesteps)-zeroat))

        axarr[plotindex].plot(timesteps, rewards)


    axarr[0, 0].legend(['pv','load','soc','cost/1000','real actions'])


    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/sample_episodes'.format(path)
    fig.savefig(path+'.png')






def evaluate(env, agent, episodes=1000, ParametricPendulumActionSpace=None):

    epsf = agent.epsilon_final
    eps = agent.epsilon
    noise = agent.noise
    agent.epsilon_final = 0.
    agent.epsilon = 0.
    agent.noise = None

    returns = []
    timesteps = []
    for _ in range(episodes):
        state = env.reset()[0]
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal and t < 200:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
      #      action = pad_action(act, act_param) #, env.pmax)
            action = toPendulumActions(act, act_param, ParametricPendulumActionSpace)
            state, reward, terminal,_, info = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)

    agent.epsilon_final = epsf
    agent.epsilon = eps
    agent.noise = noise

    return np.array(returns)









if __name__ == '__main__':
    run(episodes=5000, saveload='runPdqnaPendulum',num_sample_eps=6, loadscaling=250, seed=1, measure_step=30)
  #  run(episodes=2000, saveload='runPdqna250',num_sample_eps=6, loadscaling=250, seed=2, measure_step=30)