import gym	
import torch	
from gym.spaces import Dict	
import numpy as np	
import os	
import bauwerk	
from torch import nn	
import random	
import torch.nn.functional as F	
import collections	
from torch.optim.lr_scheduler import StepLR	
import matplotlib.pyplot as plt	
import math	
#from wrapperDiscrete_newRewardNoHindsight import wrapperDiscrete_newRewardNoHindsight	
from wrapperDiscretepg_newRewardNoHindsight import wrapperDiscrete_newRewardNoHindsight	
from bauwerk.envs.solar_battery_house import SolarBatteryHouseCoreEnv	


"""	
Implementation of Double DQN for gym environments with discrete action space.	
"""	

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")	

"""	
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.	
"""	
class QNetwork(nn.Module):	
    def __init__(self, action_dim, state_dim, hidden_dim):	
        super(QNetwork, self).__init__()	

        self.fc_1 = nn.Linear(state_dim, hidden_dim)	
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)	
        self.fc_3 = nn.Linear(hidden_dim, action_dim)	

    def forward(self, inp):	

        x1 = F.leaky_relu(self.fc_1(inp))	
        x1 = F.leaky_relu(self.fc_2(x1))	
        x1 = self.fc_3(x1)	

        return x1	


"""	
If the observations are images we use CNNs.	
"""	
class QNetworkCNN(nn.Module):	
    def __init__(self, action_dim):	
        super(QNetworkCNN, self).__init__()	

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)	
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)	
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)	
        self.fc_1 = nn.Linear(8960, 512)	
        self.fc_2 = nn.Linear(512, action_dim)	

    def forward(self, inp):	
        inp = inp.view((1, 3, 210, 160))	
        x1 = F.relu(self.conv_1(inp))	
        x1 = F.relu(self.conv_2(x1))	
        x1 = F.relu(self.conv_3(x1))	
        x1 = torch.flatten(x1, 1)	
        x1 = F.leaky_relu(self.fc_1(x1))	
        x1 = self.fc_2(x1)	

        return x1	


"""	
memory to save the state, action, reward sequence from the current episode. 	
"""	
class Memory:	
    def __init__(self, len):	
        self.rewards = collections.deque(maxlen=len)	
        self.state = collections.deque(maxlen=len)	
        self.action = collections.deque(maxlen=len)	
        self.is_done = collections.deque(maxlen=len)	

    def update(self, state, action, reward, done):	
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards	
        # and actions whcih leads to a mismatch when we sample from memory.	
        if not done:	
            self.state.append(state)	
        self.action.append(action)	
        self.rewards.append(reward)	
        self.is_done.append(done)	

    def sample(self, batch_size):	
        """	
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.	
        """	
        n = len(self.is_done)	
        idx = random.sample(range(0, n-1), batch_size)	
   #     print(n)	
   #     print(idx)	
        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)	

    def reset(self):	
        self.rewards.clear()	
        self.state.clear()	
        self.action.clear()	
        self.is_done.clear()	


def select_action(model, env, state, eps):	
    state = torch.Tensor(state).to(device)	
    with torch.no_grad():	
        values = model(state)	

    # select a random action wih probability eps	
    if random.random() <= eps:	
        action = np.random.randint(0, env.action_space.n)	
    else:	
        action = np.argmax(values.cpu().numpy())	

    return action	


def train(batch_size, current, target, optim, memory, gamma):	

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)	

    q_values = current(states)	
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)	

    next_q_values = current(next_states)	
    next_q_state_values = target(next_states)	
  	
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)	
   	
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)	

    loss = (q_value - expected_q_value.detach()).pow(2).mean()	

    optim.zero_grad()	
    loss.backward()	
    optim.step()	


def evaluate(Qmodel, env, repeats):	
    """	
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average	
    episode reward.	
    """	
    Qmodel.eval()	
    perform = 0	
    for _ in range(repeats):	

        state = env.reset()	
        state=state[0]	
        done = False	

        while not done:	

            state = torch.Tensor(state).to(device)	
            with torch.no_grad():	
                values = Qmodel(state)	

            action = np.argmax(values.cpu().numpy())	
            state, reward, done,truncated, _ = env.step(action)	
            perform += reward	

    Qmodel.train()	
    return perform/repeats	


def update_parameters(current_model, target_model):	
    target_model.load_state_dict(current_model.state_dict())	



def main(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=10, batch_size=1, update_repeats=50,	
         num_episodes=401, seed=42, max_memory_size=500000, lr_gamma=0.9, lr_step=32, measure_step=100, num_sample_eps=6, saveload='default',	
         measure_repeats=5, hidden_dim=64, env_name='bauwerk/SolarBatteryHouse-v0', cnn=False, horizon=np.inf, render=True, render_step=50, loadscaling=1):	
    """	
    :param gamma: reward discount factor	
    :param lr: learning rate for the Q-Network	
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train	
    :param eps: probability to take a random action during training	
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time	
    :param eps_min: minimal value of "eps"	
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a	
    batch of size "batch_size" from the memory.	
    :param batch_size: see above	
    :param update_repeats: see above	
    :param num_episodes: the number of episodes played in total	
    :param seed: random seed for reproducibility	
    :param max_memory_size: size of the replay memory	
    :param lr_gamma: learning rate decay for the Q-Network	
    :param lr_step: every "lr_step" episodes we decay the learning rate	
    :param measure_step: every "measure_step" episode the performance is measured	
    :param measure_repeats: the amount of episodes played in to asses performance	
    :param hidden_dim: hidden dimensions for the Q_network	
    :param env_name: name of the gym environment	
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"	
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)	
    :param render: if "True" renders the environment every "render_step" episodes	
    :param render_step: see above	
    :return: the trained Q-Network and the measured performances	
    """	
    #env = gym.make(env_name)	
    	
    cfg = { 'solar_scaling_factor' : loadscaling,	
          'load_scaling_factor' : loadscaling}	
    	
    env = gym.make("CartPole-v1") #, render_mode="rgb_array")	
    torch.manual_seed(seed)	
    	
    dataslice={}	
    dataslices=[]	
    pv_consums, maxpvs, socs, rewards, totalcosts = [], [], [], [], []	
    eptimes = np.array([])	
    epindices = np.array([])	
    eppvs = np.array([])	
    eploads = np.array([])	
    epsocs =np.array([])	
    epcosts = np.array([])	
    eprewards = np.array([])	
    sampleepisodes = []	
    countsampleepisodes = 0	
    numsampleepisodes = num_sample_eps	



    print((env.observation_space))	
    Q_1 = QNetwork(action_dim=2, state_dim=env.observation_space.shape[0],	
                                    hidden_dim=hidden_dim).to(device)	
    Q_2 = QNetwork(action_dim=2, state_dim=env.observation_space.shape[0],	
                                    hidden_dim=hidden_dim).to(device)	
    # transfer parameters from Q_1 to Q_2	
    update_parameters(Q_1, Q_2)	

    # we only train Q_1	
    for param in Q_2.parameters():	
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)	
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)	

    memory = Memory(max_memory_size)	
    performance = []


    for episode in range(num_episodes):	
        # display the performance	
        if episode % measure_step == 0:	
            performance.append([episode, evaluate(Q_1, env, measure_repeats)])	

            	
        state = env.reset()	
        state = state[0]	

        memory.state.append(state)	
        done = False	
        i = 0	
        step = 0	
        totalepreward = 0	
        while not done:	

            i += 1	
            old_state = state	
            action = select_action(Q_2, env, state, eps)	
            state, reward, done,truncated, info = env.step(action)	

            if episode % measure_step == 0 and countsampleepisodes < numsampleepisodes:	
                eptimes = np.append(eptimes,step)	

                eprewards = np.append(eprewards, reward)	
                	
            totalepreward += reward	
            if i > horizon:	
                done = True	
            # render the environment if render == True	
            if render and episode % render_step == 0:	
                env.render()	
            # save state, action, reward sequence	
            memory.update(state, action, reward, done)	
            step += 1	

        # just done	
        rewards.append(totalepreward)	
        	
     #   totalcosts.append(info["total_cost"] / 1000)	
        if episode % measure_step == 0:	
            if countsampleepisodes < numsampleepisodes:	
                sampleepisodes.append((eprewards, eptimes))
                countsampleepisodes +=1	
                eprewards, eptimes = np.array([]), np.array([])	
            if episode != 0:	
                dataslice["rewards"] = rewards	
                dataslices.append(dataslice)	
                rewards = []	

        if episode >= min_episodes and episode % update_step == 0:	
            for _ in range(update_repeats):	
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)	
            # transfer new parameter from Q_1 to Q_2	
            update_parameters(Q_1, Q_2)
	
            # update learning rate and eps	
            scheduler.step()	
            eps = max(eps*eps_decay, eps_min)	
        	
    	
    if saveload == 'default':	
        output = 'output/{}-rundef'.format(env_name)	
    else: 	
        output = 'output/{}-{}'.format(env_name,saveload)	
    namecfg = (output, batch_size, lr, seed)	
    path = '{}/bs{}lr{}seed{}'.format(*namecfg)	
    plotperformance(performance, dataslices, namecfg, path, interval=measure_step)	
    return Q_1, performance	



def plotperformance(performance, dataslices, namecfg, path, interval=None):	
    output, bsize, epsilon, seed = namecfg    	
    print(performance)	

    yrew = [slice[1] for slice in performance]	
    errorrew=[0.1 for slice in performance]	

              	
    yrewards = [np.mean(slice["rewards"]) for slice in dataslices]	
    errorrewards = [np.std(slice["rewards"]) for slice in dataslices]    	

    x = [item[0] for item in performance]	
    xless = range(interval,len(performance)*interval,interval)	
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))	
    plt.xlabel('Timestep')	
    plt.ylabel('Average Reward')	
    plt.title('epsilon: {}, batch size: {}, seed: {}'.format(epsilon, bsize, seed))	
    ax.errorbar(x, yrew, fmt='-ko')	

    ax.errorbar(xless, yrewards, fmt='-ro')	

        	
    ax.legend(['test av reward','interval av reward']) #,'interval av total costs', 'interval av SoCs','interval av pv consumption','interval av max pv consum'])	

    if not os.path.exists(path):	
        os.makedirs(path)	
    plt.savefig(path+'/validate_slices.png')	

    plt.close()	
    print("saved Average Reward")	



def plotsampleepisodeslong(data, path, loadscaling):	
    	
    numeps = len(data)	
    fig, axarr = plt.subplots(3, math.ceil(numeps/3))	
    figind, axarr2 = plt.subplots(3, math.ceil(numeps/3))	
    fig.figsize = (40, 4*math.ceil(numeps/3))	
    figind.figsize = (40, 4*math.ceil(numeps/3))	
    print(numeps)	

    for ep in range(numeps):	
        rewards, timesteps = data[ep]	
        axidb = math.floor(ep/3)	
        axida = ep - 3 * axidb	
	
        if axarr.shape == (3,):	
            plotindex = ep	
        else:	
            plotindex = (axida, axidb)	

        axarr[plotindex].plot(timesteps, rewards)	

    axarr[plotindex].legend(['reward','load','soc','cost/1000','real actions'])	

    if not os.path.exists(path):	
        os.makedirs(path)	
    fig.savefig('{}sample_episodes.png'.format(path))	

if __name__ == '__main__':	

    print(device)	