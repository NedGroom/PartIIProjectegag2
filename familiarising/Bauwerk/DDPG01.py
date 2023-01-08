# Setup and helper code
import bauwerk
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

NUM_TRAIN_STEP = 500 # 24 * 365 * 2
EVAL_LEN = 100 # 24*30 # evaluate on 1 month of actions

# Helper functions for evaluating methods

def evaluate_actions(actions, env):
    cum_reward = 0
    obs = env.reset()
    for action in actions:
        output = env.step(np.array(action, dtype=np.float32))
        obs, reward, done, _, info = output
        cum_reward += reward

    return cum_reward / len(actions)

def eval_model(model, env):
    # Obtaining model actions and evaluating them
    model_actions = []
    obs = env.reset()
    for i in range(EVAL_LEN):
        action, _states = model.predict(obs)
        model_actions.append(action)
        obs, _, _, _ = env.step(action)

    p_model = evaluate_actions(model_actions[:EVAL_LEN], env)
    if type(p_model) == np.ndarray:
        p_model = p_model[0]
    return p_model

# callback for evaluating callback during training
class EvalCallback(BaseCallback):
    def __init__(self, eval_freq = 24*7, verbose=0):
        super().__init__(verbose)
        self.data = []
        self.eval_freq = eval_freq
        self.eval_env = gym.make("bauwerk/SolarBatteryHouse-v0")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.data.append(eval_model(self.model, self.eval_env))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self.data.append(eval_model(self.model, self.eval_env))

        return True





# Create SolarBatteryHouse environment
env = gym.make("bauwerk/SolarBatteryHouse-v0")



# mean random performance over 100 trials
random_trials = [evaluate_actions([env.action_space.sample() for _ in range(EVAL_LEN)], env) for _ in range(100)]
random_std = np.std(random_trials) # standard deviation
p_rand = np.mean(random_trials)
# note: std here is standard deviation between different trials (of multiple actions)
print(type(p_rand))
print(type(random_std))
print(f"Avg reward with random actions: {p_rand:.4f} (standard deviation: {random_std:.4f})")



# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model_ddpg = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
#dqn_callback = EvalCallback()
model_ddpg.learn(total_timesteps=NUM_TRAIN_STEP, log_interval=10,progress_bar=True)
model_ddpg.save("ddpg_pendulum")
env = model_ddpg.get_env()

p_model_ddpg = eval_model(model_ddpg, env)
print(type(p_model_ddpg))
print(f"Avg reward (per step) with model actions: {p_model_ddpg:.4f}")


env = gym.make("bauwerk/SolarBatteryHouse-v0")
optimal_actions, _ = bauwerk.solve(env)
p_opt = evaluate_actions(optimal_actions[:EVAL_LEN], env)
print(f"Avg reward (per step) with optimal actions: {p_opt:.4f}")







# Measuring performance relative to random and optimal
def compute_rel_perf(p_model):
    return (p_model - p_rand)/(p_opt - p_rand)
p_rel_ddpg = compute_rel_perf(p_model_ddpg)
print(f"Performance relative to random and optimal: {p_rel_ddpg:.4f}")