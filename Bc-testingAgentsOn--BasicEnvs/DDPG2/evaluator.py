
import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.io import savemat


from util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval=1, save_path='', max_episode_length=None, args=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.epsilon = args.epsilon
        self.bsize = args.bsize
        self.seed = args.seed
        self.testrewards = np.array([]).reshape(num_episodes,0)

        self.consumslices = []
        self.maxpvslices = []
        self.socslices = []
        self.rewardslices = []

    def __call__(self, env, policy, stats={}, debug=False, save=True):

        self.stats = stats
        self.is_training = False
        observation = None
        result = []
        #env = wrapperPartial_newRewardNoHindsight( gym.make("bauwerk/SolarBatteryHouse-v0") )


        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            observation = observation[0]
            episode_reward = 0.
            episode_steps = 0.
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
                observation, reward, done, _, info = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                episode_reward += reward
                episode_steps += 1
            episode += 1
            result.append(episode_reward)

      #      if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            

        result = np.array(result).reshape(-1,1)

    #    consums = np.array(self.stats["pv_consums"]).squeeze()
    #    maxpvs = np.array(self.stats["maxpvs"]).squeeze()
    #    socs = np.array(self.stats["socs"]).squeeze()

     #   rewards = np.array(self.stats["rewards"]).squeeze()
        
        self.testrewards = np.hstack([self.testrewards, result])

     #   self.consumslices.append(np.array(consums)[None, :])
     #   self.maxpvslices.append(np.array(maxpvs)[None, :])# = np.hstack([self.maxpvslices, maxpvs])
     #   self.socslices.append(np.array(socs)[None, :])# = np.hstack([self.socslices, socs])
  #      self.rewardslices.append(np.array(rewards)[None, :])
            

        return np.mean(result), self.testrewards

    def save_results(self):
        path = '{}/bs{}eps{}seed{}'.format(self.save_path, self.bsize, self.epsilon, self.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_results_with_data('{}/validate_slices'.format(path))

    def save_results_with_data(self, fn):

        yrew = np.mean(self.testrewards, axis=0)
        errorrew=np.std(self.testrewards, axis=0)

        yconsums = [np.mean(slice) for slice in self.consumslices]
        errorconsum=[np.std(slice) for slice in self.consumslices]

        ysocs = [np.mean(slice) for slice in self.socslices]
        errorsocs = [np.std(slice) for slice in self.socslices]

        ymaxpvs = [np.mean(slice) for slice in self.maxpvslices]
        errormaxpv = [np.std(slice) for slice in self.maxpvslices]
              
        yrewards = [np.mean(slice) for slice in self.rewardslices]
        errorrewards = [np.std(slice) for slice in self.rewardslices]     

        x = range(0,self.testrewards.shape[1]*self.interval,self.interval)


        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.title('epsilon: {}, batch size: {}'.format(self.epsilon, self.bsize))
        ax.errorbar(x, yrew, yerr=errorrew, fmt='-ko')
        ax.errorbar(x, yrewards, yerr=errorrewards, fmt='-ro')
        ax.errorbar(x, ysocs, yerr=errorsocs, fmt='-bo')
        ax.errorbar(x, yconsums, yerr=errorconsum, fmt='-co')
        ax.errorbar(x, ymaxpvs, yerr=errormaxpv, fmt='c.')
        
        ax.legend(['test av reward','interval av reward','interval av SoCs','interval av pv consumption','interval av max pv consum'])

    #    ax.plot(x, ymaxpvs[None, :])
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.testrewards})
        plt.close()
        print("saved Average Reward")

