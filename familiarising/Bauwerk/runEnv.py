import bauwerk
import gym

env = gym.make("bauwerk/SolarBatteryHouse-v0")
obs = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()