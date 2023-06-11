


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
		if dataDict == {}:
			continue
		consums = dataDict['pvconsums']
		socs = dataDict['socs']
		episodes = dataDict['episodes']
		avperforms = dataDict['rewards']
	#	print(episodes)
		print(np.mean(consums, axis=0).squeeze(axis=1))
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

	print("printing std dev for consums: ", np.std(consums, axis=0).squeeze(axis=1))



def plotSearchPerformance(dataByAlg, path=None):
	fig, axarr = plt.subplots(3, 3, figsize=(10, 6))
	#axarr.set_xticks([0,500,1000,1500,2000])
	for i, (alg, dataDict) in enumerate(dataByAlg.items()):

		consums = dataDict['pvconsums']
		socs = dataDict['socs']
		episodes = dataDict['episodes']
		avperforms = dataDict['rewards']
	#	print(episodes)
		print(np.mean(consums, axis=0).squeeze(axis=1))
	#	print(np.mean(avperforms, axis=0))
	#	print(episodes)
	#	print(np.mean(avperforms, axis=0).squeeze(axis=1))
		print(alg)

		axarr[0,i].errorbar(episodes, np.mean(consums, axis=0).squeeze(axis=1), yerr=np.std(consums, axis=0).squeeze(axis=1), ecolor='lightgray',linewidth=0.3)
		axarr[1,i].errorbar(episodes, np.mean(socs, axis=0).squeeze(axis=1), yerr=np.std(socs, axis=0).squeeze(axis=1), ecolor='lightgray',linewidth=0.3)
		axarr[2,i].errorbar(episodes, np.mean(avperforms, axis=0), yerr=np.std(avperforms, axis=0), ecolor='lightgray',linewidth=0.3)

		axarr[0,i].set_xlabel("Number of episodes")
		axarr[1,i].set_xlabel("Number of episodes")
		axarr[2,i].set_xlabel("Number of episodes")

		axarr[0,i].title.set_text(alg)

	axarr[0,0].set_ylabel("PV self-consumption")
	axarr[1,0].set_ylabel("SOC at departure")
	axarr[2,0].set_ylabel("Rewards")

	plt.subplots_adjust(hspace=0.5)

	plt.savefig(path+'.png')

	print("printing std dev for consums: ", np.std(consums, axis=0).squeeze(axis=1))



def runBatchLoad(run:int, eps:int, tolerance:float, loadscaling:int):

	start_time = time.time()
	ddqndata1, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=1, update_step=30, loadscaling=loadscaling, tolerance=tolerance, num_sample_eps=6)
	ddqntime = time.time() - start_time
#	ddqndata2, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=1, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
#	ddqndata3, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=2, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
#	ddqndata4, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=3, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
#	ddqndata5, _, _ = ddqn.main(batch_size=32, measure_step=20, num_episodes=eps, saveload='masterQ'+str(run), seed=4, update_step=30, loadscaling=3000, tolerance=0.3, num_sample_eps=6)
	
	start_time = time.time()
	ddpgdata1 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=1, loadscaling = loadscaling, tolerance=tolerance)
	ddpgtime = time.time() - start_time
#	ddpgdata2 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=1, loadscaling = 3000, tolerance=0.3)
#	ddpgdata3 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=2, loadscaling = 3000, tolerance=0.3)
#	ddpgdata4 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=3, loadscaling = 3000, tolerance=0.3)
#	ddpgdata5 = ddpg.main(mode='train', train_eps=eps, warmup=15, bsize=64, saveload='masterG'+str(run), validate_every=30, validate_eps=30, epsilon=5000, seed=4, loadscaling = 3000, tolerance=0.3)


	start_time = time.time()
	pdqndata1 = pdqn.run(episodes=eps, loadscaling=loadscaling, seed=1, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6, scaleState=False, tolerance=tolerance)
	pdqntime = time.time() - start_time
#	pdqndata2 = pdqn.run(episodes=eps, loadscaling=3000, seed=1, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)
#	pdqndata3 = pdqn.run(episodes=eps, loadscaling=3000, seed=2, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)
#	pdqndata4 = pdqn.run(episodes=eps, loadscaling=3000, seed=3, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)
#	pdqndata5 = pdqn.run(episodes=eps, loadscaling=3000, seed=4, measure_step=30, saveload='masterP'+str(run), num_sample_eps=6)

	pvlens = []
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
		pvlens.append(len(data['pvconsums']))
		ddpgdata['pvconsums'].append(data['pvconsums'])
		ddpgdata['socs'].append(data['socs'])
		ddpgdata['rewards'].append(data['rewards'])
	#	print(ddpgdata)
	print(pvlens)
	timings = {'ddqn': ddqntime, 'ddpg': ddpgtime, 'pdqn': pdqntime}
	#print(timings)
	data = {'ddqn': ddqndata, 'ddpg': ddpgdata, 'pdqn': pdqndata}
	plotPerformance(data, path='output/master/'+str(run))




#runBatchLoad(run=14, eps=2000, loadscaling = 2000, tolerance=0.2)
#runBatchLoad(run=15, eps=2000, loadscaling = 2000, tolerance=0.25)
#runBatchLoad(run=16, eps=2000, loadscaling = 2000, tolerance=0.3)
#runBatchLoad(run=17, eps=2000, loadscaling = 2000, tolerance=0.35)
		
#runBatchLoad(run=18, eps=2000, loadscaling = 3400, tolerance=0.2)
#runBatchLoad(run=19, eps=2000, loadscaling = 3400, tolerance=0.25)
#runBatchLoad(run=20, eps=2000, loadscaling = 3400, tolerance=0.3)
#runBatchLoad(run=21, eps=2000, loadscaling = 3400, tolerance=0.35)

#runBatchLoad(run=22, eps=2000, loadscaling = 1000, tolerance=0.2)
#runBatchLoad(run=23, eps=2000, loadscaling = 1000, tolerance=0.3)
#runBatchLoad(run=24, eps=2000, loadscaling = 500, tolerance=0.2)
#runBatchLoad(run=25, eps=2000, loadscaling = 500, tolerance=0.3)





#runBatchLoad(run=26, eps=2000, loadscaling = 500, tolerance=0.1)
#runBatchLoad(run=27, eps=2000, loadscaling = 1000, tolerance=0.1)
#runBatchLoad(run=28, eps=2000, loadscaling = 2000, tolerance=0.1)
#runBatchLoad(run=29, eps=2000, loadscaling = 3000, tolerance=0.1)
#runBatchLoad(run=30, eps=2000, loadscaling = 3500, tolerance=0.1)

#runBatchLoad(run=31, eps=2000, loadscaling = 500, tolerance=0.1)
#runBatchLoad(run=32, eps=2000, loadscaling = 1000, tolerance=0.1)
#runBatchLoad(run=33, eps=2000, loadscaling = 2000, tolerance=0.1)
#runBatchLoad(run=34, eps=2000, loadscaling = 3500, tolerance=0.1)
#runBatchLoad(run=35, eps=2000, loadscaling = 8000, tolerance=0.1)
#runBatchLoad(run=36, eps=2000, loadscaling = 500, tolerance=0.2)
#runBatchLoad(run=37, eps=2000, loadscaling = 1000, tolerance=0.2)
#runBatchLoad(run=38, eps=2000, loadscaling = 2000, tolerance=0.2)
#runBatchLoad(run=39, eps=2000, loadscaling = 3500, tolerance=0.2)
#runBatchLoad(run=40, eps=2000, loadscaling = 8000, tolerance=0.2)


#runBatchLoad(run=31, eps=50, loadscaling = 3500, tolerance=0.1)



def exploreParams(loadscaling, tolerance, eps=10, run=None, newreward=False, infeascontrol=False):

	ddqnLR = [0.001]#0.003,0.001,0.0005]
	updateStep = [30]#5,15,30]
	ddqnParams = [ddqnLR, None, updateStep]

	ddpgLRA = [0.0001]#0.0003, 0.0001, 0.00005]
	ddpgLRC = [0.001]#0.003,0.001,0.0005]
	ddpgTAU = [0.01]#0.005, 0.01, 0.017]
	ddpgParams = [ddpgLRA, ddpgLRC, ddpgTAU]

	pdqnLRact = [0.0001]#0.0003,0.0001,0.00005]
	pdqnLRparam = [0.001]#0.003,0.001,0.0005]
	pdqnTAU = [0.001]#0.003,0.001,0.0005]
	pdqnParams = [pdqnLRact, pdqnLRparam, pdqnTAU]

	loadscaling = loadscaling
	tolerance = tolerance

	for i in range(1): #LRA
		for j in range(1): # Tau/update
			for k in range(1): # LRC
				if k == 0:
					ddqndata1, _, _ = ddqn.main(infeascontrol=infeascontrol, distanceTargetReward=newreward, lr=ddqnLR[i], update_step=updateStep[j], batch_size=32, measure_step=20, num_episodes=eps, saveload='{}/ddqn'.format(run), seed=1, loadscaling=loadscaling, tolerance=tolerance, num_sample_eps=6)
				ddpgdata1 = ddpg.main(infeascontrol=infeascontrol, distanceTargetReward=newreward, prate=ddpgLRA[i], rate=ddpgLRC[k], tau=ddpgTAU[j],mode='train', train_eps=eps, warmup=15, bsize=64, saveload='{}/ddpg'.format(run), validate_every=30, validate_eps=30, epsilon=5000, seed=1, loadscaling = loadscaling, tolerance=tolerance)
				pdqndata1 = pdqn.run(infeascontrol=infeascontrol, distanceTargetReward=newreward, learning_rate_actor=pdqnLRact[i], learning_rate_actor_param=pdqnLRparam[k], tau_actor_param=pdqnTAU[j], tau_actor=pdqnTAU[j], episodes=eps, loadscaling=loadscaling, seed=1, measure_step=30, saveload='{}/pdqn'.format(run), num_sample_eps=6, scaleState=False)

				if k==0: ddqndata = {'pvconsums':[], 'socs':[], 'rewards':[], 'episodes':ddqndata1['episodes']}
				else: ddqndata = {}
				pdqndata, ddpgdata = {'pvconsums':[], 'socs':[], 'rewards':[], 'episodes':pdqndata1['episodes']}, {'pvconsums':[], 'socs':[], 'rewards':[], 'episodes':ddpgdata1['episodes']}
				if k == 0:
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

				data = {'ddqn': ddqndata, 'ddpg': ddpgdata, 'pdqn': pdqndata}
				plotPerformance(data, path='output/master/{}/LRA{}LRC{}TAU{}'.format(run,i,k,j))


exploreParams(500, 0.3, eps=2000, run='49newrewinfeas', newreward=True, infeascontrol=True)