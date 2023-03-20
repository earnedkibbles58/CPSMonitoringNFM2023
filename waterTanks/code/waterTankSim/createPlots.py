

import os
import math
import numpy as np
import random
import scipy.stats
import matplotlib.pyplot as plt
import pickle

from parsePRISMOutput import computeViolationFromWaterLevelsKnownControl,computeViolationFromWaterLevels

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import brier_score_loss
import pandas as pd

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


def compute_ECE(binned_counts,binned_confs):

        ECE = 0
        running_data_count = 0
        for bin in binned_counts:

            bin_counts = binned_counts[bin]
            bin_confs = binned_confs[bin]
            
            if bin_counts[1] != 0:
                bin_acc = 1-bin_counts[0]/bin_counts[1]
                bin_conf = np.mean(bin_confs)

                ECE += bin_counts[1] * abs(bin_acc-bin_conf)
                running_data_count += bin_counts[1]
        
        ECE = ECE/running_data_count
        return ECE

def compute_CCE(binned_counts,binned_confs):

        CCE = 0
        running_data_count = 0
        for bin in binned_counts:

            bin_counts = binned_counts[bin]
            bin_confs = binned_confs[bin]
            
            if bin_counts[1] != 0:
                bin_acc = 1-bin_counts[0]/bin_counts[1]
                bin_conf = np.mean(bin_confs)

                if bin_conf > bin_acc:
                    CCE += bin_counts[1] * abs(bin_acc-bin_conf)
                running_data_count += bin_counts[1]
        
        CCE = CCE/running_data_count
        return CCE



def compute_ECE_perception(binned_counts,binned_confs):

        ECE = 0
        running_data_count = 0
        for bin in binned_counts:

            bin_counts = binned_counts[bin]
            bin_confs = binned_confs[bin]
            
            if bin_counts[1] != 0:
                bin_acc = bin_counts[0]/bin_counts[1]
                bin_conf = np.mean(bin_confs)

                ECE += bin_counts[1] * abs(bin_acc-bin_conf)
                running_data_count += bin_counts[1]
        
        ECE = ECE/running_data_count
        return ECE



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

    allSafeProbs = []
    allSafeProbsStateDist = []
    allTrueSafeProbs = []
    allSafeUnsafe = []

    allWL1EstVars = []
    allWL2EstVars = []
    

    allTrialLengths = []

    plotMonitorCalibration = False
    plotAUCCurve = True
    plotPerceptionCalibration = False
    plotPerceptionModel = False

    ## safety model params
    delta_wl = 1
    trimming = True
    if trimming:
        safe_prob_base_dir = "../../models/safetyProbs/knownControl/withTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(delta_wl) + "/"
        safe_prob_base_dir_unknown_control = "../../models/safetyProbs/withTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(delta_wl) + "/"
    else:
        safe_prob_base_dir = "../../models/safetyProbs/knownControl/noTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(delta_wl) + "/"
        safe_prob_base_dir_unknown_control = "../../models/safetyProbs/noTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(delta_wl) + "/"


    if trimming:
        plotSaveDir = "../../results/monitorPlots/knownControl/withTrimming/upperLimit90/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials_initControlState00_forPaper/"
    else:
        plotSaveDir = "../../results/monitorPlots/knownControl/noTrimming/upperLimit90/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials_initControlState00_forPaper/"

    trialSaveDir = plotSaveDir + "trials/"
    os.makedirs(trialSaveDir,exist_ok=True)
    trialSafeSaveDirSafe = plotSaveDir + "trialsSafe/"
    os.makedirs(trialSafeSaveDirSafe,exist_ok=True)
    trialSafeSaveDirUnsafe = plotSaveDir + "trialsUnsafe/"
    os.makedirs(trialSafeSaveDirUnsafe,exist_ok=True)
    dataSaveDir = plotSaveDir + "dataFolder/"
    os.makedirs(dataSaveDir,exist_ok=True)


    if trimming:
        percpetionDataSaveDir = "../../results/monitorPlots/knownControl/withTrimming/upperLimit90/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials_initControlState00_forPaper_stateEstimatorData/"
    else:
        percpetionDataSaveDir = "../../results/monitorPlots/knownControl/noTrimming/upperLimit90/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials_initControlState00_forPaper_stateEstimatorData/"
    perceptionCalibrationSaveDir = percpetionDataSaveDir + "dataFolder/"
    os.makedirs(dataSaveDir,exist_ok=True)

    # for j in range(numTrials):
    #     print("Trial " + str(j),flush=True)
    #     inflows = [13.5]
    #     outflows = [4.3]
        
    #     inflows_est = [12, 13, 14, 15]
    #     outflows_est = [3, 4, 5, 6]
        
        
    #     wlMax=100
    #     wlInitLow = 40
    #     wlInitHigh = 60
    #     wlInit1=random.uniform(wlInitLow,wlInitHigh)
    #     wlInit2=random.uniform(wlInitLow,wlInitHigh)
                
    #     ctrlThreshLower = 10
    #     ctrlThreshUpper = 90
        
    #     numSteps = 50
    #     numStepsPRISM = 10
    #     contAction1 = 0#random.randint(0,1)#0
    #     contAction2 = 0#random.randint(0,1)#0
        
    #     unsafe = 0

    #     mu = 0
    #     sigma = 5

    #     mu_filter = 0
    #     sigma_filter = 2
    #     # noiseDist = makedist('Normal','mu',mu,'sigma',sigma)
    #     # np.random.normal(mu,sigma,1)[0] ## SEEME: command to generate samples from noise distribution
    #     # noiseDist = scipy.stats.norm(mu,sigma)
        
    #     minValProb = 0.2
    #     maxValProb = 0.2
    #     minValProbFilter = 0.1
    #     maxValProbFilter = 0.1


    #     estimated_safety_probs = []
    #     estimated_safety_probs_state_dist = []
    #     true_safety_probs = []
    #     safe_unsafe_over_time = []

    #     estimated_safety_probs_unknown_control = []
    #     estimated_safety_probs_state_dist_unknown_control = []
    #     true_safety_probs_unknown_control = []

    #     wl1EstVars = []
    #     wl2EstVars = []

    #     wl1 = wlInit1
    #     noises1 = []
    #     wls1 = []
    #     wlPers1 = []
    #     wlEsts1 = []
    #     wlEstErrs1 = []
        
    #     wl2 = wlInit2
    #     noises2 = []
    #     wls2 = []
    #     wlPers2 = []
    #     wlEsts2 = []
    #     wlEstErrs2 = []

    #     filter_wl_disc = 1
    #     stateDist1 = []
    #     curr_wl = filter_wl_disc/2

    #     while True:
    #         if curr_wl>wlMax:
    #             break
            
    #         if curr_wl>=wlInitLow and curr_wl<=wlInitHigh:
    #             stateDist1.append(1)
    #         else:
    #             stateDist1.append(0)
            
    #         curr_wl = curr_wl + filter_wl_disc
        
    #     for i in range(len(stateDist1)):
    #         stateDist1[i] = stateDist1[i]/sum(stateDist1)
        
        
    #     stateDist2 = []
    #     curr_wl = filter_wl_disc/2
    #     while True:
    #         if curr_wl>wlMax:
    #             break
            
    #         if curr_wl>=wlInitLow and curr_wl<=wlInitHigh:
    #             stateDist2.append(1)
    #         else:
    #             stateDist2.append(0)
            
    #         curr_wl = curr_wl + filter_wl_disc
        
    #     for i in range(len(stateDist2)):
    #         stateDist2[i] = stateDist2[i]/sum(stateDist2)

    #     # wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
    #     # wlEsts1.append(wlEst1)

    #     # wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
    #     # wlEsts2.append(wlEst2)

    #     wls1.append(wl1)
    #     wls2.append(wl2)

    #     ## add short warmup phase for filter???? (e.g. like 10 steps)
    #     for i in range(numSteps):
    #         r=random.uniform(0,1)
    #         if(r<minValProb):
    #             noise1=-100
    #         elif(r>(1-maxValProb)):
    #             noise1=100
    #         else:
    #             noise1 = np.random.normal(mu,sigma,1)[0]
            
    #         noises1.append(noise1)
    #         wlPer1 = max(min(wl1+noise1,wlMax),0)
    #         wlPers1.append(wlPer1)
            
    #         stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
    #         wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
    #         wlEsts1.append(wlEst1)
            
    #         r=random.uniform(0,1)
    #         if(r<minValProb):
    #             noise2=-100
    #         elif(r>(1-maxValProb)):
    #             noise2=100
    #         else:
    #             noise2 = np.random.normal(mu,sigma,1)[0]
            
    #         noises2.append(noise2)
    #         wlPer2 = max(min(wl2+noise2,wlMax),0)
    #         wlPers2.append(wlPer2)
            
    #         stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
    #         wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
    #         wlEsts2.append(wlEst2)




    #         ## compute control tank 1
    #         if wlEst1<ctrlThreshLower or (wlEst1<ctrlThreshUpper and contAction1==1):
    #             contAction1=1
    #         else:
    #             contAction1=0
            
            
    #         ## compute control tank 2
    #         if wlEst2<ctrlThreshLower or (wlEst2<ctrlThreshUpper and contAction2==1):
    #             contAction2=1
    #         else:
    #             contAction2=0
            
            
    #         ## Global controller
    #         contActionG1=contAction1
    #         contActionG2=contAction2
    #         if(contAction1==1 and contAction2==1 and wlEst1<wlEst2):
    #             contActionG1=1
    #             contActionG2=0
    #         elif(contAction1==1 and contAction2==1 and wlEst1>wlEst2):
    #             contActionG1=0
    #             contActionG2=1
    #         elif(contAction1==1 and contAction2==1 and wlEst1 == wlEst2):
    #             r = random.uniform(0,1)
    #             if r<=0.5:
    #                 contActionG1=1
    #                 contActionG2=0
    #             else:
    #                 contActionG1=0
    #                 contActionG2=1
            
    #         ## ADDME: save water tank variance here
    #         wl1EstVar = varianceOfStateEstimate(stateDist1,filter_wl_disc)
    #         wl2EstVar = varianceOfStateEstimate(stateDist2,filter_wl_disc)

    #         wl1EstVars.append(wl1EstVar)
    #         wl2EstVars.append(wl2EstVar)

    #         ## ADDME: water tank safety using state estimate and known control
    #         wlid1 = math.ceil(max(min(wlEst1,wlMax),0)/delta_wl)
    #         wlid2 = math.ceil(max(min(wlEst2,wlMax),0)/delta_wl)
    #         tankSafeProbThisTime = 1-computeViolationFromWaterLevelsKnownControl(wlid1,wlid2,contAction1,contAction2,contActionG1,contActionG2,safe_prob_base_dir)

    #         ## ADDME: water tank safety using state estimate and unknown control
    #         wlid1 = math.ceil(max(min(wlEst1,wlMax),0)/delta_wl)
    #         wlid2 = math.ceil(max(min(wlEst2,wlMax),0)/delta_wl)
    #         tankSafeProbThisTimeUnknownControl = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir_unknown_control)

    #         ## ADDME: water tank safety using true state and known control action
    #         wlid1 = math.ceil(max(min(wl1,wlMax),0)/delta_wl)
    #         wlid2 = math.ceil(max(min(wl2,wlMax),0)/delta_wl)
    #         if wl1 <= 0 or wl2 <= 0 or wl1>wlMax or wl2>wlMax:
    #             tankTrueSafeProbThisTime=0
    #         else:
    #             tankTrueSafeProbThisTime = 1-computeViolationFromWaterLevelsKnownControl(wlid1,wlid2,contAction1,contAction2,contActionG1,contActionG2,safe_prob_base_dir)

    #         ## ADDME: water tank safety using true state and unknown control action
    #         wlid1 = math.ceil(max(min(wl1,wlMax),0)/delta_wl)
    #         wlid2 = math.ceil(max(min(wl2,wlMax),0)/delta_wl)
    #         if wl1 <= 0 or wl2 <= 0 or wl1>wlMax or wl2>wlMax:
    #             tankTrueSafeProbThisTime=0
    #         else:
    #             tankTrueSafeProbThisTimeUnknownControl = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir_unknown_control)


    #         ## water tank safety using filter distribution and known control
    #         runningSafety = 0
    #         runningProb = 0
    #         for stateInd1 in range(len(stateDist1)):
    #             for stateInd2 in range(len(stateDist2)):
    #                 state1 = filter_wl_disc*stateInd1+0.5
    #                 state2 = filter_wl_disc*stateInd2+0.5
    #                 wlid1 = math.ceil(max(min(state1,wlMax),0)/delta_wl)
    #                 wlid2 = math.ceil(max(min(state2,wlMax),0)/delta_wl)
    #                 tempSafeProbThisTime = 1-computeViolationFromWaterLevelsKnownControl(wlid1,wlid2,contAction1,contAction2,contActionG1,contActionG2,safe_prob_base_dir)
    #                 runningSafety += tempSafeProbThisTime*stateDist1[stateInd1]*stateDist2[stateInd2]
    #                 runningProb += stateDist1[stateInd1]*stateDist2[stateInd2]
    #         runningSafety = min(1,runningSafety)
    #         runningSafety = max(0,runningSafety)

    #         # print(runningProb)
    #         assert abs(runningProb-1) <= 0.0001


    #         ## water tank safety using filter distribution and unknown control
    #         runningSafetyKnownControl = 0
    #         runningProbKnownControl = 0
    #         for stateInd1 in range(len(stateDist1)):
    #             for stateInd2 in range(len(stateDist2)):
    #                 state1 = filter_wl_disc*stateInd1+0.5
    #                 state2 = filter_wl_disc*stateInd2+0.5
    #                 wlid1 = math.ceil(max(min(state1,wlMax),0)/delta_wl)
    #                 wlid2 = math.ceil(max(min(state2,wlMax),0)/delta_wl)
    #                 tempSafeProbThisTime = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir_unknown_control)
    #                 runningSafetyKnownControl += tempSafeProbThisTime*stateDist1[stateInd1]*stateDist2[stateInd2]
    #                 runningProbKnownControl += stateDist1[stateInd1]*stateDist2[stateInd2]
    #         runningSafetyKnownControl = min(1,runningSafetyKnownControl)
    #         runningSafetyKnownControl = max(0,runningSafetyKnownControl)

    #         # print(runningProb)
    #         assert abs(runningProbKnownControl-1) <= 0.0001


    #         estimated_safety_probs.append(tankSafeProbThisTime)
    #         estimated_safety_probs_state_dist.append(runningSafety)
    #         true_safety_probs.append(tankTrueSafeProbThisTime)

    #         estimated_safety_probs_unknown_control.append(tankSafeProbThisTimeUnknownControl)
    #         estimated_safety_probs_state_dist_unknown_control.append(runningSafetyKnownControl)
    #         true_safety_probs_unknown_control.append(tankTrueSafeProbThisTimeUnknownControl)
            
            
    #         wl1=wl1-random.choice(outflows)+contActionG1*random.choice(inflows)
            
    #         wls1.append(wl1)

    #         wl2=wl2-random.choice(outflows)+contActionG2*random.choice(inflows)
            
    #         wls2.append(wl2)

    #         ## update filter
    #         stateDist1 = bayesMonitorDynamics(stateDist1,contActionG1,inflows_est,outflows_est,filter_wl_disc,wlMax)
    #         wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
    #         # wlEsts1.append(wlEst1)
    #         wlEstErrs1.append(wlEst1-wl1)
    #         allPerErrs.append(wlEst1-wl1)

    #         stateDist2 = bayesMonitorDynamics(stateDist2,contActionG2,inflows_est,outflows_est,filter_wl_disc,wlMax)
    #         wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
    #         # wlEsts2.append(wlEst2)
    #         wlEstErrs2.append(wlEst2-wl2)
    #         allPerErrs.append(wlEst2-wl2)


    #         if(wl1<=0 or wl1>wlMax):
    #             unsafe=1
    #             safe_unsafe_over_time.append(1)
    #             print("Unsafe")
    #             break
    #         elif(wl2<=0 or wl2>wlMax):
    #             unsafe=1
    #             safe_unsafe_over_time.append(1)
    #             print("Unsafe")
    #             break
    #         else:
    #             safe_unsafe_over_time.append(0)

    #         # print("Safety prob: " + str(tankSafeProbThisTime))
    #         # input("Wait")
        
    #     allTrialLengths.append(len(estimated_safety_probs))

    #     # print("Trial len: " + str(allTrialLengths[-1]))

    #     unsafes=unsafes+unsafe
        
    #     for time,safeProb in enumerate(estimated_safety_probs[0:numSteps]):
    #         allSafeProbs.append(safeProb)
    #         allSafeProbsStateDist.append(estimated_safety_probs_state_dist[time])
    #         allTrueSafeProbs.append(true_safety_probs[time])

    #         allSafeProbsUnknownControl.append(estimated_safety_probs_unknown_control[time])
    #         allSafeProbsStateDistUnknownControl.append(estimated_safety_probs_state_dist_unknown_control[time])
    #         allTrueSafeProbsUnknownControl.append(true_safety_probs_unknown_control[time])

    #         allWL1EstVars.append(wl1EstVars[time])
    #         allWL2EstVars.append(wl2EstVars[time])
            
    #         if sum(safe_unsafe_over_time[time:time+numStepsPRISM]) >= 1:
    #             crash_this_time = 1
    #         else:
    #             crash_this_time = 0
    #         # if unsafe==1:
    #             # print("Time: " + str(time))
    #             # print("Crash: " + str(crash_this_time))
    #             # print("Safe prob est: " + str(safeProb))
    #             # print("Safe prob true: " + str(true_safety_probs[time]))
    #             # input("Press enter to conintue")
    #         allSafeUnsafe.append(crash_this_time)
    #         if time in safe_probs_per_time:
    #             safe_probs_per_time[time].append(safeProb)
    #             safe_probs_per_time_state_dist[time].append(estimated_safety_probs_state_dist[time])
    #             safe_unsafe_per_time[time].append(crash_this_time)
    #             true_safe_probs_per_time[time].append(true_safety_probs[time])
    #         else:
    #             safe_probs_per_time[time] = [safeProb]
    #             safe_probs_per_time_state_dist[time] = [estimated_safety_probs_state_dist[time]]
    #             safe_unsafe_per_time[time] = [crash_this_time]
    #             true_safe_probs_per_time[time] = [true_safety_probs[time]]
        
    #     if plotTrials:

    #         plt.clf()
    #         plt.plot(wls1,'r', label="Tank 1 Water Level")
    #         plt.plot(wls2,'b', label="Tank 2 Water Level")
    #         plt.plot(wlEsts1,'m', label="Tank 1 Estimated Water Level")
    #         plt.plot(wlEsts2,'g', label="Tank 2 Water Level")
    #         plt.ylim(0, wlMax)
    #         plt.xlabel("Time (s)")
    #         plt.ylabel("Water Level")
    #         plt.savefig(trialSaveDir + "/trial" + str(j) + ".png")
    #         plt.legend(loc="lower left")
    #         plt.savefig(trialSaveDir + "/trial" + str(j) + "_noLegend.png")
    #         # input("Press enter to conintue")
    #         plt.clf()

    #     ## TODO: add part saving the safety estimates over time for each trial

    #     if plotTrialsSafety:
    #         plt.clf()
    #         plt.plot(estimated_safety_probs,'b',label="State estimate monitor")
    #         plt.plot(estimated_safety_probs_state_dist,'g', label="State distribution monitor")
    #         plt.plot(true_safety_probs,'r', label="True state monitor")
    #         plt.ylim(0, 1.1)
    #         plt.xlabel("Time (s)")
    #         plt.ylabel("Safety Estimate")
    #         if unsafe == 0:
    #             plt.savefig(trialSafeSaveDirSafe + "/trialSafety" + str(j) + ".png")
    #         else:
    #             plt.savefig(trialSafeSaveDirUnsafe + "/trialSafety" + str(j) + ".png")
            
    #         plt.legend(loc="lower left")
    #         if unsafe == 0:
    #             plt.savefig(trialSafeSaveDirSafe + "/trialSafety" + str(j) + "_noLegend.png")
    #         else:
    #             plt.savefig(trialSafeSaveDirUnsafe + "/trialSafety" + str(j) + "_noLegend.png")

    #         # input("Press enter to conintue")
    #         plt.clf()


    
    ## TODO: add code here to save all the data to the folder to analyze later
    # with open(dataSaveDir + "allSafeProbs.pkl",'wb') as f:
    #     pickle.dump(allSafeProbs,f)
    # with open(dataSaveDir + "allSafeProbsStateDist.pkl",'wb') as f:
    #     pickle.dump(allSafeProbsStateDist,f)
    # with open(dataSaveDir + "allTrueSafeProbs.pkl",'wb') as f:
    #     pickle.dump(allTrueSafeProbs,f)
    # with open(dataSaveDir + "allSafeProbsUnknownControl.pkl",'wb') as f:
    #     pickle.dump(allSafeProbsUnknownControl,f)
    # with open(dataSaveDir + "allSafeProbsStateDistUnknownControl.pkl",'wb') as f:
    #     pickle.dump(allSafeProbsStateDistUnknownControl,f)
    # with open(dataSaveDir + "allTrueSafeProbsUnknownControl.pkl",'wb') as f:
    #     pickle.dump(allTrueSafeProbsUnknownControl,f)

    # with open(dataSaveDir + "allSafeUnsafe.pkl",'wb') as f:
    #     pickle.dump(allSafeUnsafe,f)
    
    # with open(dataSaveDir + "allWL1EstVars.pkl",'wb') as f:
    #     pickle.dump(allWL1EstVars,f)
    # with open(dataSaveDir + "allWL2EstVars.pkl",'wb') as f:
    #     pickle.dump(allWL2EstVars,f)

    # with open(dataSaveDir + "allPerErrs.pkl",'wb') as f:
    #     pickle.dump(allPerErrs,f)


    ## load tons of data
    with open(dataSaveDir + "allSafeProbs.pkl",'rb') as f:
        allSafeProbs = pickle.load(f)
    with open(dataSaveDir + "allSafeProbsStateDist.pkl",'rb') as f:
        allSafeProbsStateDist = pickle.load(f)
    with open(dataSaveDir + "allTrueSafeProbs.pkl",'rb') as f:
        allTrueSafeProbs = pickle.load(f)

    with open(dataSaveDir + "allSafeUnsafe.pkl",'rb') as f:
        allSafeUnsafe = pickle.load(f)
    
    with open(dataSaveDir + "allWL1EstVars.pkl",'rb') as f:
        allWL1EstVars = pickle.load(f)
    with open(dataSaveDir + "allWL2EstVars.pkl",'rb') as f:
        allWL2EstVars = pickle.load(f)

    with open(dataSaveDir + "allPerErrs.pkl",'rb') as f:
        allPerErrs = pickle.load(f)



    with open(perceptionCalibrationSaveDir + "allWL1Vals.pkl",'rb') as f:
        allWL1Vals = pickle.load(f)
    with open(perceptionCalibrationSaveDir + "allWL1Ests.pkl",'rb') as f:
        allWL1Ests = pickle.load(f)
    with open(perceptionCalibrationSaveDir + "allWL1Dists.pkl",'rb') as f:
       allWL1Dists  = pickle.load(f)


    with open(perceptionCalibrationSaveDir + "allWL2Vals.pkl",'rb') as f:
        allWL2Vals = pickle.load(f)
    with open(perceptionCalibrationSaveDir + "allWL2Ests.pkl",'rb') as f:
        allWL2Ests = pickle.load(f)
    with open(perceptionCalibrationSaveDir + "allWL2Dists.pkl",'rb') as f:
        allWL2Dists = pickle.load(f)



    ## print perception model and save data
    err_dict = dict()

    for i in range(len(allPerErrs)):
        bin_ind = round(allPerErrs[i])
        if bin_ind in err_dict:
            err_dict[bin_ind] = err_dict[bin_ind] + 1/len(allPerErrs)
        else:
            err_dict[bin_ind] = 1/len(allPerErrs)
    
    print("Perception error dict: " + str(err_dict))
        
        
    
    print('number of crashes: ' + str(unsafes) + ', ' + str(unsafes/numTrials))
    print('average length of scenario: ' + str(np.mean(allTrialLengths)))

    # allSafeProbs.sort()
    # allTrueSafeProbs.sort()

    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts = {}
    binned_counts_state_dist = {}
    binned_counts_true = {}

    binned_confs = {}
    binned_confs_state_dist = {}
    binned_confs_true = {}
    for bin in range(num_bins):
        binned_counts[bin] = [0,0]
        binned_counts_state_dist[bin] = [0,0]
        binned_counts_true[bin] = [0,0]

        binned_confs[bin] = []
        binned_confs_state_dist[bin] = []
        binned_confs_true[bin] = []

    for i in range(len(allSafeProbs)):
        conf = allSafeProbs[i]
        confStateDist = allSafeProbsStateDist[i]
        confTrue = allTrueSafeProbs[i]
        safeUnsafe = allSafeUnsafe[i] # 0 if safe, 1 if unsafe

        
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts[bin][0]+=safeUnsafe
        binned_counts[bin][1]+=1
        binned_confs[bin].append(conf)

        bin = get_bin(confStateDist,num_bins)
        binned_counts_state_dist[bin][0]+=safeUnsafe
        binned_counts_state_dist[bin][1]+=1
        binned_confs_state_dist[bin].append(confStateDist)

        bin = get_bin(confTrue,num_bins)
        binned_counts_true[bin][0]+=safeUnsafe
        binned_counts_true[bin][1]+=1
        binned_confs_true[bin].append(confTrue)




    print("Calibration across all classes conf monitor using state estimates")
    for bin in binned_counts:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts[bin][0]/binned_counts[bin][1])) + ", " + str(binned_counts[bin][1]-binned_counts[bin][0]) + "/" + str(binned_counts[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")

    print("Calibration across all classes conf monitor using state estimator distribution")
    for bin in binned_counts_state_dist:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts_state_dist[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts_state_dist[bin][0]/binned_counts_state_dist[bin][1])) + ", " + str(binned_counts_state_dist[bin][1]-binned_counts_state_dist[bin][0]) + "/" + str(binned_counts_state_dist[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")


    print("Calibration across all classes conf monitor using actual state")
    for bin in binned_counts_true:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts_true[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts_true[bin][0]/binned_counts_true[bin][1])) + ", " + str(binned_counts_true[bin][1]-binned_counts_true[bin][0]) + "/" + str(binned_counts_true[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")



    ## calibration with adaptive binning
    allSafetyProbs = []
    [allSafetyProbs.append(p) for p in allSafeProbs]
    [allSafetyProbs.append(p) for p in allSafeProbsStateDist]
    [allSafetyProbs.append(p) for p in allTrueSafeProbs]

    print("Len safety probs: " + str(len(allSafetyProbs)))
    _,safety_prob_intervals_est_state = pd.qcut(allSafetyProbs,q=num_bins,retbins=True,duplicates="drop")
    # _,safety_prob_intervals_true_state = pd.qcut(allTrueSafeProbs,q=num_bins,retbins=True)

    binned_counts_adaptive = dict()
    binned_counts_state_dist_adaptive = dict()
    binned_counts_true_adaptive = dict()

    for i in range(len(allSafeProbs)):
        conf = allSafeProbs[i]
        confStateDist = allSafeProbsStateDist[i]
        confTrue = allTrueSafeProbs[i]
        safeUnsafe = allSafeUnsafe[i] # 0 if safe, 1 if unsafe
        
        
        bin = get_bin_adaptive(conf,safety_prob_intervals_est_state)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        if bin in binned_counts_adaptive:
            binned_counts_adaptive[bin][0]+=safeUnsafe
            binned_counts_adaptive[bin][1]+=1
        else:
            binned_counts_adaptive[bin]=[safeUnsafe,1]

        bin = get_bin_adaptive(confStateDist,safety_prob_intervals_est_state)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        if bin in binned_counts_state_dist_adaptive:
            binned_counts_state_dist_adaptive[bin][0]+=safeUnsafe
            binned_counts_state_dist_adaptive[bin][1]+=1
        else:
            binned_counts_state_dist_adaptive[bin]=[safeUnsafe,1]

        bin = get_bin_adaptive(confTrue,safety_prob_intervals_est_state)
        if bin in binned_counts_true_adaptive:
            binned_counts_true_adaptive[bin][0]+=safeUnsafe
            binned_counts_true_adaptive[bin][1]+=1
        else:
            binned_counts_true_adaptive[bin]=[safeUnsafe,1]


    print("Adaptive calibration across all classes conf monitor using state estimates")
    for bin in binned_counts_adaptive:
        if binned_counts_adaptive[bin][1] != 0:
            # print(binned_counts[bin])
            print(str(bin) + ": " + str(1-float(binned_counts_adaptive[bin][0]/binned_counts_adaptive[bin][1])) + ", " + str(binned_counts_adaptive[bin][1]-binned_counts_adaptive[bin][0]) + "/" + str(binned_counts_adaptive[bin][1]))
        else:
            print(str(bin) + ": No data in this bin")

    print("Adaptive calibration across all classes conf monitor using state estimator distribution")
    for bin in binned_counts_state_dist_adaptive:
        if binned_counts_state_dist_adaptive[bin][1] != 0:
            # print(binned_counts[bin])
            print(str(bin) + ": " + str(1-float(binned_counts_state_dist_adaptive[bin][0]/binned_counts_state_dist_adaptive[bin][1])) + ", " + str(binned_counts_state_dist_adaptive[bin][1]-binned_counts_state_dist_adaptive[bin][0]) + "/" + str(binned_counts_state_dist_adaptive[bin][1]))
        else:
            print(str(bin) + ": No data in this bin")


    print("Adaptive calibration across all classes conf monitor using actual state")
    for bin in binned_counts_true_adaptive:
        if binned_counts_true_adaptive[bin][1] != 0:
            # print(binned_counts[bin])
            print(str(bin) + ": " + str(1-float(binned_counts_true_adaptive[bin][0]/binned_counts_true_adaptive[bin][1])) + ", " + str(binned_counts_true_adaptive[bin][1]-binned_counts_true_adaptive[bin][0]) + "/" + str(binned_counts_true_adaptive[bin][1]))
        else:
            print(str(bin) + ": No data in this bin")


    ## print ECE,CCE
    state_monitor_ECE = compute_ECE(binned_counts,binned_confs)
    dist_monitor_ECE = compute_ECE(binned_counts_state_dist,binned_confs_state_dist)
    true_monitor_ECE = compute_ECE(binned_counts_true,binned_confs_true)

    state_monitor_CCE = compute_CCE(binned_counts,binned_confs)
    dist_monitor_CCE = compute_CCE(binned_counts_state_dist,binned_confs_state_dist)
    true_monitor_CCE = compute_CCE(binned_counts_true,binned_confs_true)


    print("State monitor ECE: " + str(state_monitor_ECE))
    print("Dist monitor ECE: " + str(dist_monitor_ECE))
    print("True monitor ECE: " + str(true_monitor_ECE))

    print("State monitor CCE: " + str(state_monitor_CCE))
    print("Dist monitor CCE: " + str(dist_monitor_CCE))
    print("True monitor CCE: " + str(true_monitor_CCE))


    if plotMonitorCalibration:

        min_data_to_plot = 50

        print("Plotting safety vals")

        # state estimate monitor
        bars_x = []
        bars_y = []
        bars_width = []

        for bin in binned_counts:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            bin_avg = (bin_upper+bin_lower)/2

            if binned_counts[bin][1] >= min_data_to_plot:

                bars_x.append(bin_avg)
                bars_y.append(1-float(binned_counts[bin][0]/binned_counts[bin][1]))
                bars_width.append(0.1)
        

        fig, ax = plt.subplots()
        ax.bar(bars_x,bars_y,bars_width,edgecolor='k')
        plt.plot([0, 1], [0, 1], 'k--')  # perfect calibration curve
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/calibrationBarPlotStateMointor.png")

        # state disribution monitor
        bars_x = []
        bars_y = []
        bars_width = []

        for bin in binned_counts_state_dist:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            bin_avg = (bin_upper+bin_lower)/2

            if binned_counts_state_dist[bin][1] >= min_data_to_plot:

                bars_x.append(bin_avg)
                bars_y.append(1-float(binned_counts_state_dist[bin][0]/binned_counts_state_dist[bin][1]))
                bars_width.append(0.1)
        

        fig, ax = plt.subplots()
        ax.bar(bars_x,bars_y,bars_width,edgecolor='k')
        plt.plot([0, 1], [0, 1], 'k--')  # perfect calibration curve
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/calibrationBarPlotDistMointor.png")

        # true state monitor
        bars_x = []
        bars_y = []
        bars_width = []

        for bin in binned_counts_true:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            bin_avg = (bin_upper+bin_lower)/2

            if binned_counts_true[bin][1] >= min_data_to_plot:

                bars_x.append(bin_avg)
                bars_y.append(1-float(binned_counts_true[bin][0]/binned_counts_true[bin][1]))
                bars_width.append(0.1)
        

        fig, ax = plt.subplots()
        ax.bar(bars_x,bars_y,bars_width,edgecolor='k')
        plt.plot([0, 1], [0, 1], 'k--')  # perfect calibration curve
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/calibrationBarPlotTrueStateMointor.png")


        # x_vals_inst_true = []
        # y_vals_inst_true = []

        # for bin in binned_counts_true:
        #     bin_lower = bin/num_bins
        #     bin_upper = (bin+1)/num_bins
        #     bin_avg = (bin_upper+bin_lower)/2

        #     if binned_counts_true[bin][1] != 0:
        #         x_vals_inst_true.append(bin_avg)
        #         y_vals_inst_true.append(1-float(binned_counts_true[bin][0]/binned_counts_true[bin][1]))

        
        # plt.plot(x_vals_inst,y_vals_inst, 'b*')
        # plt.plot(x_vals_inst_true,y_vals_inst_true, 'r*')
        # plt.xlabel("Monitor Safety Probability")
        # plt.ylabel("Empirical Safety Probability")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.legend(["Instantaneous Classifier Confidence", "Average Classifier Confidence"])
        # plt.savefig(plotSaveDir + "/monitorOutputVEmpiricalSafety_postProcessing.png")
        # plt.clf()






    ## Compute Brier score
    y_true = [1-x for x in allSafeUnsafe] # ground truth labels
    y_probas = allSafeProbs # predicted probabilities generated by sklearn classifier
    y_probas_state_dist = allSafeProbsStateDist
    y_probs_GT_state = allTrueSafeProbs
    brier_loss = brier_score_loss(y_true, y_probas)
    brier_loss_state_dist = brier_score_loss(y_true,y_probas_state_dist)
    brier_loss_GT_state = brier_score_loss(y_true, y_probs_GT_state)
    print("Brier Score: " + str(brier_loss))
    print("Brier Score State Dist: " + str(brier_loss_state_dist))
    print("Brier Score GT State: " + str(brier_loss_GT_state))

    if plotAUCCurve:

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        roc_auc = auc(fpr, tpr)

        fpr_state_dist, tpr_state_dist, thresholds = roc_curve(y_true, y_probas_state_dist)
        roc_auc_state_dist = auc(fpr_state_dist, tpr_state_dist)

        fpr_GT_state, tpr_GT_state, thresholds_GT_state = roc_curve(y_true, y_probs_GT_state)
        roc_auc_GT_state = auc(fpr_GT_state, tpr_GT_state)


        print("Monitor AUC: " + str(roc_auc))
        print("State Dist Monitor AUC: " + str(roc_auc_state_dist))
        print("GT State AUC: " + str(roc_auc_GT_state))

        # Plot ROC curve
        plt.clf()
        plt.plot(fpr, tpr, 'b-', label='Point Estimate Monitor')
        plt.plot(fpr_state_dist, tpr_state_dist, 'g-', label='State Distribution Monitor')
        plt.plot(fpr_GT_state, tpr_GT_state, 'r-', label='True State Monitor')
        # plt.plot(fpr_unknown_control, tpr_unknown_control, 'c-', label='ROC curve (area = %0.3f)' % roc_auc_unknown_control)
        # plt.plot(fpr_state_dist_unknown_control, tpr_state_dist_unknown_control, '-', color='lightgreen' ,label='ROC curve (area = %0.3f)' % roc_auc_state_dist_unknown_control)
        # plt.plot(fpr_GT_state_unknown_control, tpr_GT_state_unknown_control, 'm-', label='ROC curve (area = %0.3f)' % roc_auc_GT_state_unknown_control)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        # plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(plotSaveDir + "/safetyMonitorAUCCurve_postProcessing.png")



    if plotPerceptionCalibration:

        num_bins = 10
        filter_wl_disc = 1
        binned_counts_perception = {}
        binned_confs_perception = {}
        for bin in range(num_bins):
            binned_counts_perception[bin] = [0,0]
            binned_confs_perception[bin] = []

        # tank 1
        for i in range(len(allWL1Vals)):
            wlEst = allWL1Ests[i]
            for j in range(len(allWL1Dists[i])):
                stateEstRange = [filter_wl_disc*j,filter_wl_disc*(j+1)]
                conf = allWL1Dists[i][j]
                if wlEst >= stateEstRange[0] and wlEst<= stateEstRange[1]:
                    wlInRange = 1
                else:
                    wlInRange = 0
                bin = get_bin(conf,num_bins)
                # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
                binned_counts_perception[bin][0]+=wlInRange
                binned_counts_perception[bin][1]+=1
                binned_confs_perception[bin].append(conf)

        # tank 2
        for i in range(len(allWL2Vals)):
            wlEst = allWL2Ests[i]
            for j in range(len(allWL2Dists[i])):
                stateEstRange = [filter_wl_disc*j,filter_wl_disc*(j+1)]
                conf = allWL2Dists[i][j]
                if wlEst >= stateEstRange[0] and wlEst<= stateEstRange[1]:
                    wlInRange = 1
                else:
                    wlInRange = 0
                bin = get_bin(conf,num_bins)
                # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
                binned_counts_perception[bin][0]+=wlInRange
                binned_counts_perception[bin][1]+=1
                binned_confs_perception[bin].append(conf)

        
        ## compute ECE
        perception_ECE = compute_ECE_perception(binned_counts_perception,binned_confs_perception)
        print("Perception ECE: " + str(perception_ECE))

        print("Calibration across all classes conf monitor using state estimates")
        for bin in binned_counts_perception:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            if binned_counts_perception[bin][1] != 0:
                # print(binned_counts[bin])
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(float(binned_counts_perception[bin][0]/binned_counts_perception[bin][1])) + ", " + str(binned_counts_perception[bin][0]) + "/" + str(binned_counts_perception[bin][1]))
            else:
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")


        ## make calibration plot
        min_data_to_plot = 0
        print("Plotting safety vals")

        # state estimate monitor
        bars_x = []
        bars_y = []
        bars_width = []

        for bin in binned_counts_perception:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            bin_avg = (bin_upper+bin_lower)/2

            if binned_counts_perception[bin][1] >= min_data_to_plot:

                bars_x.append(bin_avg)
                bars_y.append(float(binned_counts_perception[bin][0]/binned_counts_perception[bin][1]))
                bars_width.append(0.1)
        

        fig, ax = plt.subplots()
        ax.bar(bars_x,bars_y,bars_width,edgecolor='k')
        plt.plot([0, 1], [0, 1], 'k--')  # perfect calibration curve
        plt.xlabel("State Estimator Confidence")
        plt.ylabel("Empirical State Estimate Probability")
        plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/calibrationBarPlotPerception.png")


    if plotPerceptionModel:
        LECProbs = [0.0006, 0.0018, 0.0027, 0.0023, 0.0068, 0.0132, 0.0268, 0.0400, 0.0651, 0.0829, 0.1172, 0.1367, 0.1281, 0.1183, 0.0976, 0.0694, 0.0376, 0.0270, 0.0138, 0.0069, 0.0024, 0.0014, 0.0005, 0.0005, 0.0002]
        err = -12
        num_data_points = 100*50

        err_amnts = []
        bars_width = []
        counts = []
        for LECProb in LECProbs:

            err_amnts.append(err)
            bars_width.append(1)
            counts.append(LECProb*num_data_points)
            err+=1

        fig, ax = plt.subplots()
        ax.bar(err_amnts,counts,bars_width,edgecolor='k')
        plt.xlabel("State Estimator Error")
        plt.ylabel("Bin Counts")
        # plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/perceptionErrorModel.png")



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
