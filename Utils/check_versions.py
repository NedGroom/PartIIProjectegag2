import gymnasium as gym
import bauwerk
import stable_baselines3
import platform
import sys


print("Python version: " + str(sys.version))
print("Gym version: " + str(gym.__version__))
print("Bauwerk version: " + str(bauwerk.__version__))
print("StableBaselines version: " + str(stable_baselines3.__version__))

print("Python installation architecture: " + str(platform.architecture()[0]))