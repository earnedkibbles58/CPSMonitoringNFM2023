
import os
import math
import numpy as np
import random
import scipy.stats
import matplotlib.pyplot as plt
import pickle


def get_bin(conf,num_bins):

    # print(conf)
    for j in range(num_bins):
        # print((j+1)*(1/num_bins))
        if conf<(j+1)*(1/num_bins):
            return j
    return num_bins-1

def get_bin_adaptive(conf,intervals):

    for j in range(1,len(intervals)):

        bin_upper = intervals[j]
        if conf <= bin_upper:
            return (intervals[j-1],intervals[j])
    
    print("Error finding bin for conf=" + str(conf) + " and intervals: " + str(intervals))
    return -1

def main():

    # random.seed(1323)
    # random.seed(6753)
    
    seed = 23 # 100 trials upper limit 90
    # seed = 935 # 1000 trials upper limit 90


    random.seed(seed)
    np.random.seed(seed)



    numTrials = 500
    unsafes = 0
    allPerErrs = []
    
    allWL1Vals = []
    allWL2Vals = []

    allWL1Ests = []
    allWL2Ests = []

    allWL1Dists = []
    allWL2Dists = []

    ## safety model params
    delta_wl = 1
    trimming = True


    if trimming:
        plotSaveDir = "../../results/monitorPlots/knownControl/withTrimming/upperLimit90/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials_initControlState00_forPaper_stateEstimatorData/"
    else:
        plotSaveDir = "../../results/monitorPlots/knownControl/noTrimming/upperLimit90/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials_initControlState00_forPaper_stateEstimatorData/"

    dataSaveDir = plotSaveDir + "dataFolder/"
    os.makedirs(dataSaveDir,exist_ok=True)

    for j in range(numTrials):
        print("Trial " + str(j),flush=True)
        inflows = [13.5]
        outflows = [4.3]
        
        inflows_est = [12, 13, 14, 15]
        outflows_est = [3, 4, 5, 6]
        
        
        wlMax=100
        wlInitLow = 40
        wlInitHigh = 60
        wlInit1=random.uniform(wlInitLow,wlInitHigh)
        wlInit2=random.uniform(wlInitLow,wlInitHigh)
                
        ctrlThreshLower = 10
        ctrlThreshUpper = 90
        
        numSteps = 50
        numStepsPRISM = 10
        contAction1 = 0#random.randint(0,1)#0
        contAction2 = 0#random.randint(0,1)#0
        
        unsafe = 0

        mu = 0
        sigma = 5

        mu_filter = 0
        sigma_filter = 2
        
        minValProb = 0.2
        maxValProb = 0.2
        minValProbFilter = 0.1
        maxValProbFilter = 0.1

        wl1 = wlInit1
        wl2 = wlInit2

        filter_wl_disc = 1
        stateDist1 = []
        curr_wl = filter_wl_disc/2

        while True:
            if curr_wl>wlMax:
                break
            
            if curr_wl>=wlInitLow and curr_wl<=wlInitHigh:
                stateDist1.append(1)
            else:
                stateDist1.append(0)
            
            curr_wl = curr_wl + filter_wl_disc
        
        for i in range(len(stateDist1)):
            stateDist1[i] = stateDist1[i]/sum(stateDist1)
        
        
        stateDist2 = []
        curr_wl = filter_wl_disc/2
        while True:
            if curr_wl>wlMax:
                break
            
            if curr_wl>=wlInitLow and curr_wl<=wlInitHigh:
                stateDist2.append(1)
            else:
                stateDist2.append(0)
            
            curr_wl = curr_wl + filter_wl_disc
        
        for i in range(len(stateDist2)):
            stateDist2[i] = stateDist2[i]/sum(stateDist2)


        ## add short warmup phase for filter???? (e.g. like 10 steps)
        for i in range(numSteps):
            r=random.uniform(0,1)
            if(r<minValProb):
                noise1=-100
            elif(r>(1-maxValProb)):
                noise1=100
            else:
                noise1 = np.random.normal(mu,sigma,1)[0]
            
            wlPer1 = max(min(wl1+noise1,wlMax),0)
            
            stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
            wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
            
            r=random.uniform(0,1)
            if(r<minValProb):
                noise2=-100
            elif(r>(1-maxValProb)):
                noise2=100
            else:
                noise2 = np.random.normal(mu,sigma,1)[0]
            
            wlPer2 = max(min(wl2+noise2,wlMax),0)
            
            stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
            wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)


            ## TODO: save state estimator data here
            allWL1Vals.append(wl1)
            allWL1Ests.append(wlEst1)
            allWL1Dists.append(stateDist1)

            allWL2Vals.append(wl2)
            allWL2Ests.append(wlEst2)
            allWL2Dists.append(stateDist2)

            allPerErrs.append(wlEst1-wl1)
            allPerErrs.append(wlEst2-wl2)

            ## compute control tank 1
            if wlEst1<ctrlThreshLower or (wlEst1<ctrlThreshUpper and contAction1==1):
                contAction1=1
            else:
                contAction1=0
            
            
            ## compute control tank 2
            if wlEst2<ctrlThreshLower or (wlEst2<ctrlThreshUpper and contAction2==1):
                contAction2=1
            else:
                contAction2=0
            
            
            ## Global controller
            contActionG1=contAction1
            contActionG2=contAction2
            if(contAction1==1 and contAction2==1 and wlEst1<wlEst2):
                contActionG1=1
                contActionG2=0
            elif(contAction1==1 and contAction2==1 and wlEst1>wlEst2):
                contActionG1=0
                contActionG2=1
            elif(contAction1==1 and contAction2==1 and wlEst1 == wlEst2):
                r = random.uniform(0,1)
                if r<=0.5:
                    contActionG1=1
                    contActionG2=0
                else:
                    contActionG1=0
                    contActionG2=1
                        
            wl1=wl1-random.choice(outflows)+contActionG1*random.choice(inflows)
            wl2=wl2-random.choice(outflows)+contActionG2*random.choice(inflows)
            

            ## update filter
            stateDist1 = bayesMonitorDynamics(stateDist1,contActionG1,inflows_est,outflows_est,filter_wl_disc,wlMax)
            stateDist2 = bayesMonitorDynamics(stateDist2,contActionG2,inflows_est,outflows_est,filter_wl_disc,wlMax)

            if(wl1<=0 or wl1>wlMax):
                unsafe=1
                print("Unsafe")
                break
            elif(wl2<=0 or wl2>wlMax):
                unsafe=1
                print("Unsafe")
                break

        unsafes=unsafes+unsafe
        

    print("Saving data to folder " + str(dataSaveDir))
    ## TODO: add code here to save all the data to the folder to analyze later
    with open(dataSaveDir + "allWL1Vals.pkl",'wb') as f:
        pickle.dump(allWL1Vals,f)
    with open(dataSaveDir + "allWL1Ests.pkl",'wb') as f:
        pickle.dump(allWL1Ests,f)
    with open(dataSaveDir + "allWL1Dists.pkl",'wb') as f:
        pickle.dump(allWL1Dists,f)


    with open(dataSaveDir + "allWL2Vals.pkl",'wb') as f:
        pickle.dump(allWL2Vals,f)
    with open(dataSaveDir + "allWL2Ests.pkl",'wb') as f:
        pickle.dump(allWL2Ests,f)
    with open(dataSaveDir + "allWL2Dists.pkl",'wb') as f:
        pickle.dump(allWL2Dists,f)



    with open(dataSaveDir + "allPerErrs.pkl",'wb') as f:
        pickle.dump(allPerErrs,f)

    print("Num unsafes: " + str(unsafes))
    print("Prop unsafes: " + str(unsafes/numTrials))


def wlEstFromStateDist(stateDist,wlDisc):

    wlEst = 0
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist[i]
        curr_wl = wlDisc*i+0.5
        wlEst = wlEst + curr_wl_prob*curr_wl
    
    return wlEst




def bayesMonitorDynamics(stateDist,controlCommand,inflows,outflows,wlDisc,wlMax):

    newStateDist = np.zeros(len(stateDist))
    
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist[i]
        curr_wl = wlDisc*i+0.5
        
        for j in range(len(inflows)):
            
            for k in range(len(outflows)):
                next_wl = curr_wl+inflows[j]*controlCommand-outflows[k]
                next_wl = max(next_wl,0)
                next_wl = min(next_wl,wlMax)
                bin_ind = min(getBin(next_wl,wlDisc),len(stateDist)-1)
                newStateDist[bin_ind] = newStateDist[bin_ind] + curr_wl_prob*(1/len(inflows))*(1/len(outflows))
            
        
        
    
    
    
    return newStateDist




def bayesMonitorPerception(stateDist,wlReading,mu,sigma,wlDisc,minValProb,maxValProb,wlMax):

    
    newStateDist = np.zeros(len(stateDist))
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist[i]
        curr_wl = wlDisc*i+0.5
        prob = probOfReading(curr_wl,wlReading,mu,sigma,minValProb,maxValProb,wlMax)
        newStateDist[i] = curr_wl_prob*prob
        
    
    stateDist = newStateDist/sum(newStateDist)
    return stateDist



def getBin(wl,wlDisc):

    bin = wl-(wl%wlDisc)
    return int(bin)


def probOfReading(wl,wlReading,mu,sigma,minValProb,maxValProb,wlMax):
    
    probability = 0
    if wlReading == 0:
        probability = minValProb
    elif wlReading == wlMax:
        probability = maxValProb

    noiseDist = scipy.stats.norm(mu,sigma)
    
    probability = probability + (1-minValProb-maxValProb)*noiseDist.pdf(wlReading-wl)
    return probability


def varianceOfStateEstimate(stateDist,wlDisc):

    # compute mean
    wlEst = 0
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist[i]
        curr_wl = wlDisc*i+0.5
        wlEst = wlEst + curr_wl_prob*curr_wl

    wlVar = 0
    for i in range(len(stateDist)):
        curr_wl_prob = stateDist[i]
        curr_wl = (wlDisc*i+0.5 - wlEst)**2
        wlVar = wlVar + curr_wl_prob*curr_wl



if __name__ == '__main__':
    main()
