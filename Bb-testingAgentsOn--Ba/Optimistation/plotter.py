
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

class Plotter:
	def __init__(self):
		self.steps = []
		self.timeofday = []
		self.loads = []
		self.pvs = []
		self.EVPVs = []
		self.EVGs = []
		self.SOCs = []
		self.newEpisodeAt = []


	def observeinfo(self, step, info):
		self.steps.append(step)
		self.timeofday.append(info["time_step"])
		self.loads.append(info["load"])
		self.pvs.append(info["pv_gen"])
		realaction = info["realcharge_action"]
		if realaction < 0:
			self.EVPVs.append([0])
			self.EVGs.append(-realaction)
		else:
			self.EVPVs.append(realaction)
			self.EVGs.append([0])
		self.SOCs.append(info["battery_cont"])

	def startepisode(self, step):
		self.newEpisodeAt.append(step)

	def getData(self):
		return {
			"steps" : self.steps,
			"timeofday" : self.timeofday,
			"loads" : self.loads,
			"pvs" : self.pvs,
			"EVPVs" : self.EVPVs,
			"EVGs" : self.EVGs,
			"SOCs" : self.SOCs,
			"newEpisodeAt" : self.newEpisodeAt 
			}


	def saveData(self, path, seed, name, run):
		
		steps = self.steps,
		timeofday = self.timeofday,
		loads = self.loads,
		pvs = self.pvs,
		EVPVs = self.EVPVs,
		EVGs = self.EVGs,
		SOCs = self.SOCs,
		newEpisodeAt = self.newEpisodeAt 

		path = 'Output/SavedData/{}/{}'.format(path,seed)
		if not os.path.exists(path):
			os.makedirs(path)
		path = '{}/savename={}-run{}'.format(path, name, run)

			
		np.savez(path, steps = steps,
						timeofday = timeofday,
						loads = loads,
						pvs = pvs,
						EVPVs = EVPVs,
						EVGs = EVGs,
						SOCs = SOCs,
						newEpisodeAt = newEpisodeAt )
		print("data saved at: ")
		print(path)

		totaldata = {
			"times" : self.steps,
			"timeofday" : self.timeofday,
			"loads" : self.loads,
			"pvs" : self.pvs,
			"EVPVs" : self.EVPVs,
			"EVGs" : self.EVGs,
			"SOCs" : self.SOCs,
			"newEpisodeAt" : self.newEpisodeAt 
			}


		with open(path, 'wb') as fp:
			pickle.dump(totaldata, fp)
			print('dictionary saved successfully to file')
			print(path)



	def plotSolutionsFromEp(self, path,eps,name):

		with open(path, 'rb') as fp:
			datas = pickle.load(fp)

		plots = len(datas)
		fig, ax = plt.subplots(plots + 1, 1)



		for idx, data in enumerate(datas):
			print("new plot")
			

			steps = data["times"]
			timeofday = data["timeofday"]
			loads = data["loads"]
			pvs = data["pvs"]
			evpvs = data["EVPVs"]
			evgs = data["EVGs"]
			socs = data["SOCs"]

			ax[0].plot(steps, pvs, label="PV")
			ax[0].plot(steps, loads, label="Load")
			ax[idx].set_xticks([])
		
			ax[idx+1].bar(steps, evpvs / 3700, color='blue', label="EVpv")
			ax[idx+1].bar(steps, evgs / 3700, color='red', label="EVg")
			ax[idx+1].plot(steps, socs, label="SOC")


		plt.legend(bbox_to_anchor=(0., 3.52, 1., .02), loc=8,
				ncol=5, mode="expand", borderaxespad=0.)
		plt.xticks(steps[::3], timeofday[::3], rotation=45)
		plt.savefig('output/Data/{}eps/plot2sols-{}.png'.format(eps, name))
		plt.close()


	def plotEpsFromPaths(self, paths,eps,name):

		plots = len(paths)
		fig, ax = plt.subplots(plots + 1, 1)

		for idx, path in enumerate(paths):
			with open(path, 'rb') as fp:
				data = pickle.load(fp)

				steps = data["times"]
				timeofday = data["timeofday"]
				loads = data["loads"]
				pvs = data["pvs"]
				evpvs = np.array(data["EVPVs"]).squeeze() / 3700
				evgs = np.array(data["EVGs"]).squeeze() / 3700
				print(evgs)
				socs = data["SOCs"]

				ax[0].plot(steps, pvs, label="PV")
				ax[0].plot(steps, loads, label="Loads")
				ax[idx].set_xticks([])

				ax[idx+1].bar(steps, evpvs, color='blue', label="EVPV")
				ax[idx+1].bar(steps, evgs, color='red', label="EVG")
				ax[idx+1].plot(steps, socs, label="SOC")

		xticks = [str(int(x%24))+".00" for x in timeofday]
		plt.xticks(steps[::3], xticks[::3], rotation=45)
		plt.legend()
		plt.savefig('output/Data/{}eps/plot-{}.png'.format(eps, name))
		plt.close()


plotter = Plotter()

paths = ['output/Data/3eps/detopt-run1.pkl', 'output/SavedData/MPC/123/savename=firsttry-run0']
plotter.plotEpsFromPaths(paths, eps=3, name="hi")

#path = 'output/Data/1eps/detopt2sols-run4.pkl'   # path for the 1 ep 2 solutions thing
#plotter.plotSolutionsFromEp(path,eps=1,name="two-solutions")

