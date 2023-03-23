#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import sys

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight

#gym.undo_logger_setup()

def train(num_episodes, agent, env,  evaluate, validate_every, output, debug=False, warmup=0):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    
    while episode < num_episodes:
        # reset if it is the start of episode
        if done or observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
            episode += 1

        # agent pick action ...
        if step <= warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env step, observe, and update policy
        observation, reward, done, _, info = env.step(action)
        agent.observe(reward, observation, done)
        if step > warmup :
            agent.update_policy()

        # [optional] evaluate
        if evaluate is not None and validate_every > 0 and episode % validate_every == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)


        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=False, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


def main(mode='', train_iter=0, bsizze=64, epsilon=50000, validate_eps=20, validate_every=1000, seed=1, warmup=0, saveload='default'):

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
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=500, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
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
    args.train_iter = train_iter
    args.bsize=bsizze
    args.epsilon = epsilon
    args.validate_episodes = validate_eps
    args.validate_every = validate_every
    args.seed = seed
    args.warmup = warmup
    self.saveload = saveload
 #   env = NormalizedEnv(gym.make(args.env))
    env = wrapperPartial_newRewardNoHindsight(gym.make(args.env))


    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    num_states = len(env.observation_space)
    num_actions = env.action_space.shape[0]

    agent = DDPG(num_states, num_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.saveload)


    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, args.validate_every, args.saveload, 
              warmup=args.warm, updebug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.saveload,
             visualize=False, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


if __name__ == '__main__':
    main(mode='train')  