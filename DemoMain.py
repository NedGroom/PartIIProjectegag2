
#	This document is a first draft for demonstration of proficiency in using PyTorch for running a neural network architecture.
#
#	Components of basic structure: (for labelled data)
#		1. Prepare data
#			a. Load data from sources
#			b. Reformat data
#		2. Prepare dataloader
#			a. Suitable initialisation function for dataloader given format of incoming data
#			b. Identify parameters such as: batch size, shuffle, etc
#		3. Class definition for the neural/algorithm model
#		4. Evaluation from model outputs
#			a. Reformat output data
#			b. Functions for producing graphs for quantitative analysis of performance
#
#		Main loop
#			a. Load data into dataloader
#			b. Train the model on training data
#			c. Evaluate the model on test data
	
#	For reinforcement learning, we clearly do not have labelled data, but we can simply plot the reward against time.

