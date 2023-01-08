import gymnasium as gym


env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

print(gym.envs.registry.keys())

for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()