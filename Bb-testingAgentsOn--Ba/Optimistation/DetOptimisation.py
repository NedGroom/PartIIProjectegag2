import gurobipy as gp
from gurobipy import GRB
import glob
import numpy as np
from termcolor import colored
import pickle






def findTarrs(times):
    tarr = []
    for idx, time in enumerate(times):
        if idx == 0:
            tarr.append(time)
        elif time - times[idx-1] != 1 and time != 24 and time != 0:
                tarr.append(time)
    return tarr




#################

def runOptimisation(steps, predictedloads, predictedpvs, startcharge):

    pmax = 3700   
    Cbatt = 16000

    timesteps = len([i for i in range(steps)])

    # Create a new model
    m = gp.Model("EVHouse")

    # Create variables

    tdep = timesteps -1 

    Lt = m.addVars(timesteps, lb=0.0, vtype=GRB.CONTINUOUS, name="L[t]")  
    LPVt = m.addVars(timesteps, lb=0.0, vtype=GRB.CONTINUOUS, name="LPV[t]")  
    LGt = m.addVars(timesteps, lb=0.0, vtype=GRB.CONTINUOUS, name="LG[t]")   

    PVt = m.addVars(timesteps, lb=0.0, vtype=GRB.CONTINUOUS, name="PV[t]")  

    EVt = m.addVars(timesteps, lb=0.0, ub=pmax, vtype=GRB.CONTINUOUS, name="EV[t]")      # charge power
    EVPVt = m.addVars(timesteps, lb=0.0, ub=pmax, vtype=GRB.CONTINUOUS, name="EVPV[t]")    
    EVGt = m.addVars(timesteps, lb=0.0, ub=pmax, vtype=GRB.CONTINUOUS, name="EVG[t]")   

    GBt = m.addVars(timesteps, lb=0.0, vtype=GRB.CONTINUOUS, name="PB[t]")    # power bought
    GSt = m.addVars(timesteps, lb=0.0, vtype=GRB.CONTINUOUS, name="PS[t]")    # power sold

    SOCt = m.addVars(timesteps, lb=0.2, ub=1.0, vtype=GRB.CONTINUOUS, name="SOC[t]")  
    atmptSOCt = m.addVars(timesteps, lb=0.2, ub=2.0, vtype=GRB.CONTINUOUS, name="attmptSOC[t]")  

    pvconsum = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="PVconsum")
    endSOC = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="endSOC")


    # Set objective
    m.setObjective( pvconsum + endSOC , GRB.MAXIMIZE)


    # Add constraints:
    m.addConstr( SOCt[0] == startcharge , "get start battery state")

    for t in range(timesteps): #,i:

        #  m.addConstr( GSt[t] == 0 , "Enforce not selling to grid")

        m.addConstr( PVt[t]  == predictedpvs[t] , "load data")
        m.addConstr( Lt[t]  == predictedloads[t] , "load data")

    #    m.addConstr( PVt[t] + GBt[t]  == EVt[t] + Lt[t] + GSt[t] ,  "conseravtion of power") # PB is power from grid, PS is power sold to grid, yG (in 0,1) is var that forbids simultaneous purchasing and selling from/to grid
        m.addConstr( EVt[t] == EVGt[t] + EVPVt[t]               ,   "EV charged from grid and solar") 
        m.addConstr( Lt[t] == LGt[t] + LPVt[t] ,                    "Conservation of load")            # load satisfaction eq(1)
        m.addConstr( GBt[t] == EVGt[t] + LGt[t] ,                   "Amount bought from grid")
        m.addConstr( GSt[t] == PVt[t] - LPVt[t] - EVPVt[t] ,        "Amount sold to grid")

        m.addConstr( EVPVt[t] * EVGt[t] == 0 ,                      "Cannot charge car from PV and grid")
        m.addConstr( GSt[t] * GBt[t] == 0 ,                         "Cannot sell and buy from grid at same time")
        #  m.addConstr( GSt[t] * EVGt[t] == 0 ,                        "If selling to grid, then cannot be charging car from grid")

        if t != timesteps-1:
            m.addConstr( atmptSOCt[t+1] == SOCt[t] + (EVt[t])/Cbatt ,    "attempted state of charge")        # SOC continuity eq(7)
            m.addConstr( SOCt[t+1] == gp.min_(atmptSOCt, 1) ,    "charge the EV")        # SOC continuity eq(7)
        
    m.addConstr( pvconsum * PVt.sum() == (EVPVt.sum() + LPVt.sum()),"Calculating pvconsum")
    m.addConstr( endSOC == SOCt[tdep] ,                             "extracting final SOC")

    orignumvars = m.NumVars
    m.Params.nonconvex=2

    # Optimize model
    m.optimize()

    

    if m.status == GRB.OPTIMAL:

        solutions = []
        print("n solitions found: ", m.SolCount)

        for n in range(m.SolCount):
            m.setParam(GRB.Param.SolutionNumber, n)

            vars = m.getVars()[orignumvars:]

            stepsarr = [ i for i in range(steps)] 
            loads = [ v.Xn for v in vars if v.varname.startswith('L[t]')]
            pvs = [ v.Xn for v in vars if v.varname.startswith('PV[t]')]
            EVPVs = [ v.Xn for v in vars if v.varname.startswith('EVPV[t]')]
            EVGs = [ v.Xn for v in vars if v.varname.startswith('EVG[t]')]
            SOCs = [ v.Xn for v in vars if v.varname.startswith('SOC[t]')]

            print([ v.varname for v in vars if v.varname.startswith('SOC[t]')])


            solutions.append( {
                "steps" : stepsarr,
			    "loads" : loads,
			    "pvs" : pvs,
			    "EVPVs" : EVPVs,
			    "EVGs" : EVGs,
			    "SOCs" : SOCs,
			    } )

        return solutions


    elif m.status == GRB.INFEASIBLE:
        
        relaxedobjective = m.feasRelaxS(2, True, False, True)

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print("\nSlack values:")
            slacks = m.getVars()[orignumvars:]
            artVarsByName = [ v.VarName for v in slacks if v.varname.startswith('Art') ]
            #print(artVarsByName)
            artVarsByValue = [ v.VarName for v in slacks if v.X > 1e-9 ]
            print(artVarsByValue)

        print(colored("had to relax",'red'))

        return {}
   

def loadDataRunOpts(loadscale=3000,solarscale=3000,seed=1000,episodes=3, run=0):

    path = 'Utils/TestData/exampleTestData-runTest1/loadscale{}solarscale{}seed{}wrappedTrueeps{}'.format(
                                            loadscale,solarscale,seed,episodes)
    files = glob.glob(path + '*.npz')
    print("files: ", files)
    datasets = []
    for file in files:
        data = np.load(file, allow_pickle=True)
        datasets.append(data)
    testset = datasets[0]


    SOCarr = testset['SOCarr'] # get from testDataSimulation
    loads = testset['loads']
    pvs = testset['pvs']
    times = testset['times']
    tarr_ = testset['epstartsteps']

    totaltimes, totaltimesofday, totalloads, totalpvs = np.array([]), np.array([]), np.array([]), np.array([])
    totalEVPVs, totalEVGs, totalSOCs = np.array([]), np.array([]), np.array([])


    for idx, start in enumerate(tarr_):
        if idx != len(tarr_)-1:
            end = tarr_[idx+1]
            steps = end - start
        else:
            end = None
            steps = len(pvs) - start
        data = runOptimisation(steps
                                , pvs[start:end]
                                , loads[start:end]
                                , SOCarr[idx]/16000)
        data = data[0]
        totaltimes = np.append(totaltimes,times[start:end]) 
        print(data["steps"])
    #    assert False
        totaltimesofday = np.append(totaltimesofday,times[start:end]) 
        totalloads = np.append(totalloads,data["loads"]) 
        totalpvs = np.append(totalpvs,data["pvs"]) 
        totalEVPVs = np.append(totalEVPVs,data["EVPVs"]) 
        totalEVGs = np.append(totalEVGs,data["EVGs"]) 
        totalSOCs = np.append(totalSOCs,data["SOCs"]) 


    totaldata = {
                "times" : totaltimes,
                "timeofday": totaltimesofday,
			    "loads" : totalloads,
			    "pvs" : totalpvs,
			    "EVPVs" : totalEVPVs,
			    "EVGs" : totalEVGs,
			    "SOCs" : totalSOCs
			    }

    path = 'output/Data/{}eps/detopt-run{}.pkl'.format(episodes, run)
    with open(path, 'wb') as fp:
        pickle.dump(totaldata, fp)
        print('dictionary saved successfully to file')
        print(path)





loadDataRunOpts(episodes=3, run=1)









#print("times: ", timesteps)
#print("loads: ", loads)
#print("pvs: ", pvs)
#print("soc: ", SOCarr)


#m.addConstr( SOCt[0] == SOCarr[0] , "get start battery state")#

#for t in timesteps: #,i:

 #   m.addConstr( yGt[t] == 1 , "Enforce not selling to grid")

 #   m.addConstr( PVt[t]  == testset['pvs'][t] , "c")
 #   m.addConstr( Lt[t]  == testset['loads'][t] , "c, for timestep "+str(t))

 #   m.addConstr( PVt[t] + yGt[t]*PBt[t]  == EVt[t] + Lt[t] + (1-yGt[t])*PSt[t], "conseravtion of power, for timestep "+str(t)) # PB is power from grid, PS is power sold to grid, yG (in 0,1) is var that forbids simultaneous purchasing and selling from/to grid
 #   m.addConstr( PVt[t] == EVPVt[t] + LPVt[t] + PSt[t] , "conservation of PV gen, for timestep "+str(t))
 #   m.addConstr( PBt[t] == EVGt[t] + LGt[t] , "Amount bought from grid, for timestep "+str(t))
 #   m.addConstr( EVt[t] == yEVt[t]*EVGt[t] + (1-yEVt[t])*EVPVt[t] , "EV charged from grid or solar, for timestep "+str(t)) #yEVt governs simultaneity of power supply to EV according to eq(2) EVG*EVPV=0, such as:
 #   m.addConstr( yEVt[t] <= yGt[t] , "If selling to grid, then cannot be charging car from grid, for timestep "+str(t))
 #   m.addConstr( Lt[t] == LGt[t] + LPVt[t] , "Conservation of load, for timestep "+str(t))    # load satisfaction eq(1)
 ## #  m.addConstr( 0.2 <= SOCt[t] <= 1 , "c, for timestep "+str(t))    # SOC limits eq(5)
 # ##  m.addConstr( EVGt[t] <= pmax , "c, for timestep "+str(t))    # EV charging power limit eq(6)
 ###   m.addConstr( EVPVt[t] <= pmax , "c, for timestep "+str(t))
#for t in timesteps[:-1]:
#    m.addConstr( SOCt[t+1] == SOCt[t] + (EVGt[t] + EVPVt[t])/Cbatt , "charge the EV, for timestep "+str(t))        # SOC continuity eq(7)
        
#m.addConstr( pvconsum * PVt.sum() == (EVPVt.sum() + LPVt.sum())  , "Calculating pvconsum")
#m.addConstr( endSOC == SOCt[tdep] , "extracting final SOC")

#yGt = m.addVars(timesteps, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="yG[t]")   # if == 0, selling to grid. If == 1, then buying from grid.
#yEVt = m.addVars(timesteps, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="yEV[t]")  # if = 1, then charging from grid, if = 0 then charging from solar
