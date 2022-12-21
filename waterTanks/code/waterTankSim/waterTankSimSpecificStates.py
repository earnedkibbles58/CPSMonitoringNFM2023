

import os
import math
import numpy as np
import random
import scipy.stats
import matplotlib.pyplot as plt


from parsePRISMOutput import computeViolationFromWaterLevels

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

def main():

    # random.seed(1323)
    # random.seed(6753)
    random.seed(23)
    np.random.seed(23)

    numTrials = 100
    allPerErrs = []


    plotEachTrial = False
    plotAUCCurve = False
    plotTrials = False

    initWL1 = 6
    initWL2 = 8
    initControl1 = 0
    initControl2 = 0
    perceptionWarmUpSteps = 10
    perceptionInitRange = 10


    ## safety model params
    delta_wl = 1
    trimming = True
    if trimming:
        safe_prob_base_dir = "../../models/safetyProbs/withTrimming/if13.5_of4.3_deltawl" + str(delta_wl) + "/"
    else:
        safe_prob_base_dir = "../../models/safetyProbs/noTrimming/if13.5_of4.3_deltawl" + str(delta_wl) + "/"


    if trimming:
        plotSaveDir = "../../results/monitorPlots/specificStates/withTrimming/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials/wl1_" + str(initWL1) + "_wl2_" + str(initWL2) + "_contAction1_" + str(initControl1) + "_contAction2_" + str(initControl2) + "/"
    else:
        plotSaveDir = "../../results/monitorPlots/specificStates/noTrimming/deltawl_" + str(delta_wl) + "_" + str(numTrials) + "Trials/wl1_" + str(initWL1) + "_wl2_" + str(initWL2) + "_contAction1_" + str(initControl1) + "_contAction2_" + str(initControl2) + "/"

    trialSaveDir = plotSaveDir + "trials/"
    print(plotSaveDir)
    os.makedirs(trialSaveDir,exist_ok=True)

    safeProbs = []
    trueSafeProbs = []
    safeUnsafes = []


    for j in range(numTrials):
        print("Trial " + str(j),flush=True)
        inflows = [13.5]
        outflows = [4.3]
        
        inflows_est = [12, 13, 14, 15]
        outflows_est = [3, 4, 5, 6]
        
        
        wlMax=100
        wlInit1=initWL1
        wlInit2=initWL2
                
        ctrlThreshLower = 10
        ctrlThreshUpper = 85
        
        numStepsPRISM = 10
        contAction1 = initControl1
        contAction2 = initControl2
        
        unsafe = 0

        mu = 0
        sigma = 5

        mu_filter = 0
        sigma_filter = 2
        # noiseDist = makedist('Normal','mu',mu,'sigma',sigma)
        # np.random.normal(mu,sigma,1)[0] ## SEEME: command to generate samples from noise distribution
        # noiseDist = scipy.stats.norm(mu,sigma)
        
        minValProb = 0.2
        maxValProb = 0.2
        minValProbFilter = 0.1
        maxValProbFilter = 0.1

        wl1 = wlInit1
        noises1 = []
        wls1 = []
        wlPers1 = []
        wlEsts1 = []
        wlEstErrs1 = []
        
        wl2 = wlInit2
        noises2 = []
        wls2 = []
        wlPers2 = []
        wlEsts2 = []
        wlEstErrs2 = []

        filter_wl_disc = 1
        stateDist1 = []
        curr_wl = filter_wl_disc/2

        while True:
            if curr_wl>wlMax:
                break
            
            if curr_wl>=wl1-perceptionInitRange and curr_wl<=wl1+perceptionInitRange:
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
            
            if curr_wl>=wl2-perceptionInitRange and curr_wl<=wl2+perceptionInitRange:
                stateDist2.append(1)
            else:
                stateDist2.append(0)
            
            curr_wl = curr_wl + filter_wl_disc
        
        for i in range(len(stateDist2)):
            stateDist2[i] = stateDist2[i]/sum(stateDist2)

        # wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
        # wlEsts1.append(wlEst1)

        # wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
        # wlEsts2.append(wlEst2)

        # wls1.append(wl1)
        # wls2.append(wl2)

        for i in range(perceptionWarmUpSteps):
            r=random.uniform(0,1)
            if(r<minValProb):
                noise1=-100
            elif(r>(1-maxValProb)):
                noise1=100
            else:
                noise1 = np.random.normal(mu,sigma,1)[0]
            wlPer1 = max(min(wl1+noise1,wlMax),0)

            r=random.uniform(0,1)
            if(r<minValProb):
                noise2=-100
            elif(r>(1-maxValProb)):
                noise2=100
            else:
                noise2 = np.random.normal(mu,sigma,1)[0]
            wlPer2 = max(min(wl2+2,wlMax),0)

            stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
            stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)

        wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
        wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)

        # print("True state: " + str([wl1,wl2]))
        # print("Est state: " + str([wlEst1,wlEst2]))
        # input("wait")
        

        ## ADDME: compute water tank safety here
        wlid1 = math.ceil(max(min(wlEst1,wlMax),0)/delta_wl)
        wlid2 = math.ceil(max(min(wlEst2,wlMax),0)/delta_wl)
        tankSafeProb = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir)

        ## ADDME: compute water tank safety here
        wlid1 = math.ceil(max(min(wl1,wlMax),0)/delta_wl)
        wlid2 = math.ceil(max(min(wl2,wlMax),0)/delta_wl)
        if wl1 <= 0 or wl2 <= 0 or wl1>wlMax or wl2>wlMax:
            tankTrueSafeProb=0
        else:
            tankTrueSafeProb = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir)

        safeProbs.append(tankSafeProb)
        trueSafeProbs.append(tankTrueSafeProb)

        for i in range(numStepsPRISM):
            r=random.uniform(0,1)
            if(r<minValProb):
                noise1=-100
            elif(r>(1-maxValProb)):
                noise1=100
            else:
                noise1 = np.random.normal(mu,sigma,1)[0]
            
            # noises1.append(noise1)
            wlPer1 = max(min(wl1+noise1,wlMax),0)
            # wlPers1.append(wlPer1)
            
            stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
            wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
            # wlEsts1.append(wlEst1)
            
            r=random.uniform(0,1)
            if(r<minValProb):
                noise2=-100
            elif(r>(1-maxValProb)):
                noise2=100
            else:
                noise2 = np.random.normal(mu,sigma,1)[0]
            
            # noises2.append(noise2)
            wlPer2 = max(min(wl2+noise2,wlMax),0)
            # wlPers2.append(wlPer2)
            
            stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,mu_filter,sigma_filter,filter_wl_disc,minValProbFilter,maxValProbFilter,wlMax)
            wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
            # wlEsts2.append(wlEst2)

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
            # wls1.append(wl1)

            wl2=wl2-random.choice(outflows)+contActionG2*random.choice(inflows)
            # wls2.append(wl2)

            ## update filter
            stateDist1 = bayesMonitorDynamics(stateDist1,contActionG1,inflows_est,outflows_est,filter_wl_disc,wlMax)
            wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
            # wlEsts1.append(wlEst1)
            # wlEstErrs1.append(wlEst1-wl1)
            allPerErrs.append(wlEst1-wl1)

            stateDist2 = bayesMonitorDynamics(stateDist2,contActionG2,inflows_est,outflows_est,filter_wl_disc,wlMax)
            wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
            # wlEsts2.append(wlEst2)
            # wlEstErrs2.append(wlEst2-wl2)
            allPerErrs.append(wlEst2-wl2)


            if(wl1<=0 or wl1>wlMax):
                unsafe=1
                print("Unsafe")
                break
            elif(wl2<=0 or wl2>wlMax):
                unsafe=1
                print("Unsafe")
                break

            # print("Safety prob: " + str(tankSafeProbThisTime))
            # input("Wait")
        
        safeUnsafes.append(unsafe)


        
        
        if plotTrials:

            plt.clf()
            plt.plot(wls1,'r')
            plt.plot(wls2,'b')
            plt.plot(wlEsts1,'m')
            plt.plot(wlEsts2,'g')
            plt.ylim(0, wlMax)
            plt.savefig("tempTrialPlotzState" + str(initWL1) + "_" + str(initWL2) + "_" + str(initControl1) + str(initControl2) + ".png")
            # input("Press enter to conintue")
            plt.clf()

    unsafes = sum(safeUnsafes)


    ##
    print("number of crashes: " + str(unsafes))
    print("Percetage safe: " + str(1-unsafes/numTrials))
    print("True state safe: " + str(trueSafeProbs[0]))
    print("Average est state safe: " + str(sum(safeProbs)/numTrials))
    # print("Monitor safe ests: " + str(safeProbs))

    

    # plt.hist(allPerErrs,bins=10)
    # plt.savefig("testTankSimPython.png")
    # plt.clf()

    # ## print perception model and save data
    # err_dict = dict()

    # for i in range(len(allPerErrs)):
    #     bin_ind = round(allPerErrs[i])
    #     if bin_ind in err_dict:
    #         err_dict[bin_ind] = err_dict[bin_ind] + 1/len(allPerErrs)
    #     else:
    #         err_dict[bin_ind] = 1/len(allPerErrs)
    
    # print("Perception error dict: " + str(err_dict))
        
        
    
    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts = {}
    binned_counts_true = {}
    for bin in range(num_bins):
        binned_counts[bin] = [0,0]
        binned_counts_true[bin] = [0,0]


    for i in range(len(safeProbs)):
        conf = safeProbs[i]
        confTrue = trueSafeProbs[i]
        safeUnsafe = safeUnsafes[i] # 0 if safe, 1 if unsafe
        
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts[bin][0]+=safeUnsafe
        binned_counts[bin][1]+=1

        bin = get_bin(confTrue,num_bins)
        binned_counts_true[bin][0]+=safeUnsafe
        binned_counts_true[bin][1]+=1


    # print("Calibration across all classes conf monitor using state estimates")
    # for bin in binned_counts:
    #     bin_lower = bin/num_bins
    #     bin_upper = (bin+1)/num_bins
    #     if binned_counts[bin][1] != 0:
    #         # print(binned_counts[bin])
    #         print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts[bin][0]/binned_counts[bin][1])) + ", " + str(binned_counts[bin][1]-binned_counts[bin][0]) + "/" + str(binned_counts[bin][1]))
    #     else:
    #         print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")

    # print("Calibration across all classes conf monitor using actual state")
    # for bin in binned_counts_true:
    #     bin_lower = bin/num_bins
    #     bin_upper = (bin+1)/num_bins
    #     if binned_counts_true[bin][1] != 0:
    #         # print(binned_counts[bin])
    #         print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts_true[bin][0]/binned_counts_true[bin][1])) + ", " + str(binned_counts_true[bin][1]-binned_counts_true[bin][0]) + "/" + str(binned_counts_true[bin][1]))
    #     else:
    #         print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")


    ## calibration with adaptive binning
    allSafetyProbs = []
    [allSafetyProbs.append(p) for p in safeProbs]
    [allSafetyProbs.append(p) for p in trueSafeProbs]

    _,safety_prob_intervals_est_state = pd.qcut(allSafetyProbs,q=num_bins,retbins=True,duplicates="drop")
    # _,safety_prob_intervals_true_state = pd.qcut(allTrueSafeProbs,q=num_bins,retbins=True)

    binned_counts_adaptive = dict()
    binned_counts_true_adaptive = dict()

    for i in range(len(safeProbs)):
        conf = safeProbs[i]
        confTrue = trueSafeProbs[i]
        safeUnsafe = safeUnsafes[i] # 0 if safe, 1 if unsafe
        
        
        bin = get_bin_adaptive(conf,safety_prob_intervals_est_state)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        if bin in binned_counts_adaptive:
            binned_counts_adaptive[bin][0]+=safeUnsafe
            binned_counts_adaptive[bin][1]+=1
        else:
            binned_counts_adaptive[bin]=[safeUnsafe,1]


        bin = get_bin_adaptive(confTrue,safety_prob_intervals_est_state)
        if bin in binned_counts_true_adaptive:
            binned_counts_true_adaptive[bin][0]+=safeUnsafe
            binned_counts_true_adaptive[bin][1]+=1
        else:
            binned_counts_true_adaptive[bin]=[safeUnsafe,1]

    # print("Adaptive calibration across all classes conf monitor using state estimates")
    # for bin in binned_counts_adaptive:
    #     if binned_counts_adaptive[bin][1] != 0:
    #         # print(binned_counts[bin])
    #         print(str(bin) + ": " + str(1-float(binned_counts_adaptive[bin][0]/binned_counts_adaptive[bin][1])) + ", " + str(binned_counts_adaptive[bin][1]-binned_counts_adaptive[bin][0]) + "/" + str(binned_counts_adaptive[bin][1]))
    #     else:
    #         print(str(bin) + ": No data in this bin")

    # print("Adaptive calibration across all classes conf monitor using actual state")
    # for bin in binned_counts_true_adaptive:
    #     if binned_counts_true_adaptive[bin][1] != 0:
    #         # print(binned_counts[bin])
    #         print(str(bin) + ": " + str(1-float(binned_counts_true_adaptive[bin][0]/binned_counts_true_adaptive[bin][1])) + ", " + str(binned_counts_true_adaptive[bin][1]-binned_counts_true_adaptive[bin][0]) + "/" + str(binned_counts_true_adaptive[bin][1]))
    #     else:
    #         print(str(bin) + ": No data in this bin")

    ## Compute Brier score
    y_true = [1-x for x in safeUnsafes] # ground truth labels
    y_probas = safeProbs # predicted probabilities generated by sklearn classifier
    y_probs_GT_state = trueSafeProbs
    brier_loss = brier_score_loss(y_true, y_probas)
    brier_loss_GT_state = brier_score_loss(y_true, y_probs_GT_state)
    # print("Brier Score: " + str(brier_loss))
    # print("Brier Score GT State: " + str(brier_loss_GT_state))

    if plotAUCCurve:

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        roc_auc = auc(fpr, tpr)

        fpr_GT_state, tpr_GT_state, thresholds_GT_state = roc_curve(y_true, y_probs_GT_state)
        roc_auc_GT_state = auc(fpr_GT_state, tpr_GT_state)


        # Plot ROC curve
        plt.clf()
        plt.plot(fpr, tpr, 'b-', label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot(fpr_GT_state, tpr_GT_state, 'r-', label='ROC curve (area = %0.3f)' % roc_auc_GT_state)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(plotSaveDir + "/safetyMonitorAUCCurve.png")



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





if __name__ == '__main__':
    main()
