

#######

# this works:



import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.io import savemat
from wrapperPartial_newRewardNoHindsight import wrapperPartial_newRewardNoHindsight


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
        self.rewards = np.array([]).reshape(num_episodes,0)
        print(self.rewards)
        self.consumslices = []
        self.maxpvslices = []
        self.socslices = []

    def __call__(self, env, policy, stats={}, debug=False, save=True):

        self.stats = stats
        self.is_training = False
        observation = None
        result = []
        env = wrapperPartial_newRewardNoHindsight( gym.make("bauwerk/SolarBatteryHouse-v0") )


        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_reward = 0.
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
                observation, reward, done, _, info = env.step(action)
                episode_reward += reward
            episode += 1
            result.append(episode_reward)

      #      if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            

        result = np.array(result).reshape(-1,1)
        consums = np.array(self.stats["pv_consums"]).squeeze()
        print("hi")
        print(consums)
        print(consums[None, :])
        maxpvs = np.array(self.stats["maxpvs"]).squeeze()
        socs = np.array(self.stats["socs"]).squeeze()
        
        self.rewards = np.hstack([self.rewards, result])
   #     self.consumslices = np.hstack([self.consumslices, consums])
        print(self.consumslices)
        #self.consumslices = np.concatenate((self.consumslices,np.array(consums)[None, :]), axis=0)
        self.consumslices.append(np.array(consums)[None, :])
        print(self.consumslices)
        self.maxpvslices = np.hstack([self.maxpvslices, maxpvs])
        self.socslices = np.hstack([self.socslices, socs])

        if save:
            path = '{}/bs{}eps{}seed{}'.format(self.save_path, self.bsize, self.epsilon, self.seed)
            if not os.path.exists(path):
                os.makedirs(path)
            self.save_results('{}/validate_slices'.format(path))

        return np.mean(result)

    def save_results(self, fn):

        yrew = np.mean(self.rewards, axis=0)
        errorrew=np.std(self.rewards, axis=0)
    #    print(len(list) for list in self.consumslices)
        yconsums = [np.mean(slice) for slice in self.consumslices]
    #    print(self.consumslices)
        print(yconsums)
        errorconsum=[np.std(slice) for slice in self.consumslices]
        ymaxpvs = np.mean(self.maxpvslices, axis=0)
        ysocs = np.mean(self.socslices, axis=0)
        errorsocs = np.std(self.socslices, axis=0)
                    
        x = range(0,self.rewards.shape[1]*self.interval,self.interval)
        print(len(x))
       # print(self.results)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.title('epsilon: {}, batch size: {}'.format(self.epsilon, self.bsize))
        ax.errorbar(x, yrew, yerr=errorrew, fmt='-o')
        ax.errorbar(x, yconsums, yerr=errorconsum, fmt='-o')
  #      ax.errorbar(x, ysocs, yerr=errorsocs, fmt='-o')
    #    ax.plot(x, ymaxpvs[None, :])
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.rewards})
        print("saved Average Reward")