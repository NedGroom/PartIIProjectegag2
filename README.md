﻿# PartIIProjectegag2

This is the readme that was submitted with the code for my dissertation at Computer Lab, University of Cambridge, 2023.


## Code written

Files that were written completely from scratch by me include all wrapper files for the new environments, the formulation for deterministic optimisation and MPC and RBC, Util tools, and plotting functions used on: ``RL algorithms, the deterministic optimisation and MPC benchmarks, and the naive, and rule-based control techniques''.

Wrapper example files (while there may be lots of duplication) include Ba../phase4/newBaseEnvWrapper.py, Ba../phase4/discreteOverBase.py, Ba../phase3/wrapperParaAgent_newRewardNoHindsight.py.

The only code that was not written is the contructor and class files for the imported learning agents. These are described and located below.
In Ab, Bb, Bc, and Ca there include folders for applying the learning agent to an environement.

- Folders named DDPG and DDQN contain implementations of those algorithms that did not work and so was not worked on for more than a couple of hours, so there is 
only very marginal amount of my own code included here.

- In folders named DDPG2, DDQN2, PDQN, the only code that is my own is contained in the following files: DDPG2/main.py, DDPG2/evaluator.py, DDQN2/DDQN_discrete.py, PDQN/pdqn_copy_cut.py. (note that there are multiple versions of these at the different development stages)
These files contained the skeleton code for the training and limited testing loops, and this is where I added lots of code for data collection and visualisation, as well as environment tests.




## Results - the below graphs are found in the FinalGraphs folder in the repository.

Here I will describe the images of plotted results and graphs that I have managed to acquire.

- 0. Fixed dataflow so that the cycles stay regular after episode time jumps.
- 1. Example graphs from searching hyper-parameter space of the learning agents on the environment.
	(overview of different runs, an example run showing the 3 agents together, and then this example run shown individally for each agent)
	(finally a graph showing the best parameter set found, examples given for DDPG and PDQN.)
- 2. Training graphs of the learning agents of the basic environments that are easier to learn.
	(DDQN on Cartpole, DDPG on Pendulum, PDQN on Parametric Pendulum)

- 3. Plots of training performance on HER.
- 3. Plots of training performance with new reward.
- 3. Plots of training performance with new penalty for overcharging.

- 4. Plots of test performance for naive, random, rbc (rule-based control)
- 5. Test performance for deterministic optimisation


Here is a general summary for the evaluation:
- Deterministic optimisation is the peak performance baseline, but cannot be used online, only retrospectively.
- MPC is an adaptation which can be used online but uses huge compute resources and time so is inefficient.
- RBC performs very well but complex rules need to be maintained, so it is not good for scaling to a different size or structure of problem.
- Deep reinforcement learning control is supposedly superior to the alternatives as, once trained, it performs very efficiently online and manages to prioritise the aims.
	- It is able to scale much better to complex problems.


Here is a general summary for the conclusion:
- Use of reinforcement learning algorithms on a new application can be tricky as they are very sensitive to setup and initialisation parameters.
- For emulating the paper:
	- They did not provide any of their code or data source: my data generation is not distributed very similarly to their examples.
	- The episode structure they used involves the car leaving uniformly randomly between 7am and 5pm which is not a very realistic scenareo. A learning agent would be able to learn more easily if the EV movements were more consistent, ie closer to a traditional commuting timeline.
