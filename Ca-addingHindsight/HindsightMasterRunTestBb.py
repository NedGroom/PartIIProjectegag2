


import PDQN.pdqn_copy_cut as pdqn
import DDPG2.main as ddpg
import DDQN2.DDQN_discrete as ddqn
#from PDQN import pdqn_copy_cut
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import time



def plotPerformance(dataByAlg, path=None):
	fig, axarr = plt.subplots(3, 3, figsize=(10, 6))
	#axarr.set_xticks([0,500,1000,1500,2000])
	for i, (alg, dataDict) in enumerate(dataByAlg.items()):
		consums = dataDict['pvconsums']
		socs = dataDict['socs']
		episodes = dataDict['episodes']
		avperforms = dataDict['rewards']
	#	print(episodes)
#		print(np.mean(consums, axis=0).squeeze(axis=1))
	#	print(np.mean(avperforms, axis=0))
	#	print(episodes)
	#	print(np.mean(avperforms, axis=0).squeeze(axis=1))
		print(alg)

		axarr[0,i].errorbar(episodes, np.mean(consums, axis=0).squeeze(axis=1), yerr=np.std(consums, axis=0).squeeze(axis=1), ecolor='lightgray')
		axarr[1,i].errorbar(episodes, np.mean(socs, axis=0).squeeze(axis=1), yerr=np.std(socs, axis=0).squeeze(axis=1), ecolor='lightgray')
		axarr[2,i].errorbar(episodes, np.mean(avperforms, axis=0), yerr=np.std(avperforms, axis=0), ecolor='lightgray')

		axarr[0,i].set_xlabel("Number of episodes")
		axarr[1,i].set_xlabel("Number of episodes")
		axarr[2,i].set_xlabel("Number of episodes")

		axarr[0,i].title.set_text(alg)

	axarr[0,0].set_ylabel("PV self-consumption")
	axarr[1,0].set_ylabel("SOC at departure")
	axarr[2,0].set_ylabel("Rewards")

	plt.subplots_adjust(hspace=0.5)

	plt.savefig(path+'.png')



def runBatchLoad(run:int, eps:int, loadscaling:int, tolerance:float):

	start_time = time.time()
	ddqndata1, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=2, saveload='masterQ'+str(run), seed=1, update_step=30, loadscaling=loadscaling, tolerance=tolerance, num_sample_eps=6)
	ddqntime = time.time() - start_time
	#ddqndata2, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=2, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
	#ddqndata3, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=3, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
	#ddqndata4, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=4, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
	#ddqndata5, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=5, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
	
	start_time = time.time()
	ddpgdata1 = ddpg.main(mode='train', train_eps=30, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=loadscaling, seed=5, loadscaling = 3000, tolerance=tolerance)
	ddpgtime = time.time() - start_time
	#ddpgdata2 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=1, loadscaling = 3000, tolerance=0.3)
	#ddpgdata3 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=2, loadscaling = 3000, tolerance=0.3)
	#ddpgdata4 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=3, loadscaling = 3000, tolerance=0.3)
	#ddpgdata5 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=4, loadscaling = 3000, tolerance=0.3)


	start_time = time.time()
	pdqndata1 = pdqn.run(episodes=eps, scaleState=True, loadscaling=loadscaling, seed=5, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6, tolerance=tolerance)
	pdqntime = time.time() - start_time
	#pdqndata2 = pdqn.run(episodes=eps, loadscaling=3000, seed=1, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)
	#pdqndata3 = pdqn.run(episodes=eps, loadscaling=3000, seed=2, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)
	#pdqndata4 = pdqn.run(episodes=eps, loadscaling=3000, seed=3, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)
	#pdqndata5 = pdqn.run(episodes=eps, loadscaling=3000, seed=4, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)

	ddqndata, pdqndata, ddpgdata = {'pvconsums':[], 'socs':[], 'rewards':[], 'episodes':ddqndata1['episodes']}, {'pvconsums':[], 'socs':[], 'rewards':[], 'episodes':pdqndata1['episodes']}, {'pvconsums':[], 'socs':[], 'rewards':[], 'episodes':ddpgdata1['episodes']}
	for data in [ddqndata1]:#,ddqndata2,ddqndata3,ddqndata4,ddqndata5]:
		ddqndata['pvconsums'].append(data['pvconsums'])
		ddqndata['socs'].append(data['socs'])
		ddqndata['rewards'].append(data['rewards'])
	for data in [pdqndata1]:#,pdqndata2,pdqndata3,pdqndata4,pdqndata5]:
		pdqndata['pvconsums'].append(data['pvconsums'])
		pdqndata['socs'].append(data['socs'])
		pdqndata['rewards'].append(data['rewards'])
	for data in [ddpgdata1]:#,ddpgdata2,ddpgdata3,ddpgdata4,ddpgdata5]:
		ddpgdata['pvconsums'].append(data['pvconsums'])
		ddpgdata['socs'].append(data['socs'])
		ddpgdata['rewards'].append(data['rewards'])

	timings = {'ddqn': ddqntime, 'ddpg': ddpgtime, 'pdqn': pdqntime}
	print(timings)
	data = {'ddqn': ddqndata, 'ddpg': ddpgdata, 'pdqn': pdqndata}
	plotPerformance(data, path='output/masterhind/'+str(run))




#runBatchLoad(run=16, eps=2000, loadscaling=500, tolerance=0.1)
#runBatchLoad(run=17, eps=2000, loadscaling=500, tolerance=0.2)
#runBatchLoad(run=18, eps=2000, loadscaling=500, tolerance=0.3)
#runBatchLoad(run=19, eps=2000, loadscaling=1000, tolerance=0.1)
#runBatchLoad(run=20, eps=2000, loadscaling=1000, tolerance=0.2)
#runBatchLoad(run=21, eps=2000, loadscaling=1000, tolerance=0.3)
runBatchLoad(run=22, eps=100, loadscaling=1000, tolerance=0.3)
		