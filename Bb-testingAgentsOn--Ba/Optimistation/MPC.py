import gurobipy as gp
from gurobipy import GRB
import glob
import numpy as np
from termcolor import colored
from bauwerk.envs.solar_battery_house import SolarBatteryHouseEnv
from newBaseEnvWrapper import NewBaseEnvWrapper
from plotter import Plotter
from DetOptimisation import runOptimisation

def loaddata():

    path = 'Utils/TestData/exampleTestData-runTest1/loadscale3000solarscale3000seed1000wrappedTrueeps1'
    files = glob.glob(path + '*.npz')
    print("files: ", files)
    datasets = []
    for file in files:
        data = np.load(file, allow_pickle=True)
        datasets.append(data)
    testset = datasets[0]


    def findTarrs(times):
        tarr = []
        for idx, time in enumerate(times):
            if idx == 0:
                tarr.append(int(time))
            elif time - times[idx-1] != 1 and time != 24 and time != 0:
                    tarr.append(int(time))
        return tarr


    print(testset)
    tarr_ = findTarrs(testset['times'])


    SOCarr = testset['SOCarr'] # get from testDataSimulation
    loads = testset['loads']
    pvs = testset['pvs']
    times = testset['times']
    numtimes = len(times)
    startcharge = SOCarr[0]

  #  assert loads[step] == obs[0] and pvs[step] == obs[1]
    return (tarr_, loads, pvs)



def runMPC(path, testseed, name, run, episodes):

    cfg = { 'solar_scaling_factor' : 3000,
          'load_scaling_factor' : 3000} 
    env = SolarBatteryHouseEnv( cfg )
    env = NewBaseEnvWrapper( env, tolerance=0.3 , seed=testseed)
    plotter = Plotter()
    step = 0
    tarr_, loads, pvs = loaddata()

    print(env.cfg.obs_keys)

    for episode in range(episodes):

        obs = env.reset(seed=testseed)
        done = False
        plotter.startepisode(step)
        tarr = tarr_[episode]

        while not done:
            step += 1

            startcharge = obs[2]
            stepsleft = int(obs[3])

          #  loadmodel.observe(obs[0])
          #  pvmodel.observe(obs[1])

         #   predictedloads = loadmodel.predict(stepsleft)
         #   predictedpvs = pvmodel.predict(stepsleft)
            print(tarr+stepsleft)
            predictedloads = loads[step:int(tarr+stepsleft)]
            predictedpvs = pvs[step:int(tarr+stepsleft)]



            results = runOptimisation(stepsleft, predictedloads, predictedpvs, startcharge)
            if results == {}:
                action = np.array([0.], dtype=np.float32)
            else:
                action = results[0]['EVPGs'][0] - results[0]['EVGs'][0]
                nextSOC = results[0]['SOCs'][1]

            obs, reward, done, _, info = env.step(action)
            plotter.observeinfo(step, info)

            if results != {}:
                assert obs[2] == nextSOC

    
    data = plotter.saveData("MPC", 123, "firsttry", 0)


runMPC(path="MPC", testseed=123, name="firsttry", run=0, episodes=1)




