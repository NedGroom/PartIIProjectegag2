import os
import sys
#sys.path.append('C:/Users/ngroo/OneDrive/Documents/_cambridge-work/II/Part II Project/gitrepo1/Utils/MPDQNmaster')
#sys.path.append('C:/Users/ngroo/OneDrive/Documents/_cambridge-work/II/Part II Project/gitrepo1')

import click
import time
import gym
import bauwerk
#import gym_platform
#from gym.wrappers import Monitor

#import SBHparametricWrapperforAgent1
from SBHparametricWrapperforAgent1 import ParametricActions
from wrapperParaAgent_newRewardNoHindsight import wrapperParaAgent_newRewardNoHindsight



import numpy as np


#from MPDQNmaster import common, agents
from agents.pdqn import PDQNAgent
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

from agents.pdqn_multipass import MultiPassPDQNAgent



import click
import ast


class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)


@click.command()
@click.option('--seed', default=4, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=2, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=100, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--layers', default='[128,]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--multipass', default=False, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=False, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--title', default="PDDQN", help="Prefix of output files", type=str)
#

def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        use_ornstein_noise, replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, num_episodes=None):

    if num_episodes is not None:
        episodes = num_episodes

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (save_frames and visualise)
    

    env = gym.make('bauwerk/SolarBatteryHouse-v0')
    env = wrapperParaAgent_newRewardNoHindsight(env)
    print("pre-Observation space: ")
    print(env.observation_space)
    print("pre-Action space: ")
    print(env.action_space)
    initial_params_ = [3., 10., 400.]

    # env = ScaledStateWrapper(env)     why do i need to scale the observation space
 #   env = PlatformFlattenedActionWrapper(env)
#    if scale_actions:
 #       env = ScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir,title)
  #  env = Monitor(env, directory=os.path.join(dir,str(seed)), video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    print("Observation space: ")
    print(env.observation_space)
    print("Action space: ")
    print(env.action_space)



    assert not (multipass)
    agent_class = PDQNAgent
    if multipass:
        print("miltipass")
        assert 0
        agent_class = MultiPassPDQNAgent
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
        print("obs space.spaces")
        print(env.observation_space.spaces)
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.__len__()))
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
    # agent.epsilon_final = 0.
    # agent.epsilon = 0.
    # agent.noise = None

    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))
    #    print("reset")
        obs = env.reset()
   #     print(obs)
  #      print(state)
  #      print(state.values())
 #       obs = list(obs.values())
 #       print(state)
        state = [i[0] for i in obs]
        state = np.array(state, dtype=np.float32, copy=False)
    #    print(state)
  #      print(state)
        act, act_param, all_action_parameters = agent.act(state)
   #     print("padding action")
     #   print(act)
     #   print(act_param)
     #   print(all_action_parameters)
        action = pad_action(act, act_param, env.pmax)
     #   print(action)
   #      print(action)

        episode_reward = 0.
        agent.start_episode()
        terminal=False
        for j in range(max_steps):

            ret = env.step(action)
            (next_state, steps), reward, terminal, truncated, _ = ret
            #next_state = list(next_state.values())
            next_state = [i[0] for i in next_state]
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            print(next_act_param)
            next_action = pad_action(next_act, next_act_param, env.pmax)
            agent.step(state, (act, all_action_parameters), reward, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            
            action = next_action
            state = next_state

            episode_reward += reward
            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                #print("terminal")
              #  env.reset()
                break
        agent.end_episode()


        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 100 == 0: # 100 episodes
            print(str(i))
            print("Total average return: " + str(total_reward / (i + 1)))
            print("Average return over last 100: " + str(np.array(returns[-100:]).mean()))
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

    np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, 5) #evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)






def pad_action(act, act_param, pmax):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    print(act_param)
    params[act] = min(act_param, np.array([pmax - 0.1], dtype=np.float32))
    return (act, params[0], params[1])


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)









if __name__ == '__main__':
    run()