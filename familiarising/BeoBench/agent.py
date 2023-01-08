from beobench.experiment.provider import create_env, config

# create environment and get starting observation
env = create_env()
observation = env.reset()

for _ in range(config["agent"]["config"]["num_steps"]):
    # sample random action from environment's action space
    action = env.action_space.sample()
    # take selected action in environment
    observation, reward, done, info = env.step(action)

env.close()