

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


def get_bin(conf,num_bins):

    # print(conf)
    for j in range(num_bins):
        # print((j+1)*(1/num_bins))
        if conf<(j+1)*(1/num_bins):
            return j
    return num_bins-1



def main():

    random.seed(1323)

    numTrials = 100
    unsafes = 0
    allPerErrs = []

    allSafeProbs = []
    allTrueSafeProbs = []
    allSafeUnsafe = []

    safe_probs_per_time = {}
    safe_unsafe_per_time = {}
    true_safe_probs_per_time = {}

    allTrialLengths = []

    plotMonitorCalibration = True
    plotEachTrial = False
    plotAUCCurve = True

    ## safety model params
    delta_wl = 2
    trimming = True
    if trimming:
        safe_prob_base_dir = "../../models/safetyProbs/withTrimming/if13.5_of4.3_deltawl" + str(delta_wl) + "/"
    else:
        safe_prob_base_dir = "../../models/safetyProbs/noTrimming/if13.5_of4.3_deltawl" + str(delta_wl) + "/"


    if trimming:
        plotSaveDir = "../../results/monitorPlots/withTrimming/deltawl_" + str(delta_wl) + "/"
    else:
        plotSaveDir = "../../results/monitorPlots/noTrimming/deltawl_" + str(delta_wl) + "/"

    trialSaveDir = plotSaveDir + "trials/"
    os.makedirs(trialSaveDir,exist_ok=True)


    for j in range(numTrials):
        print("Trial " + str(j),flush=True)
        inflows = [13.5]
        outflows = [4.3]
        
        inflows_est = [12, 13, 14, 15]
        outflows_est = [3, 4, 5, 6]
        
        
        wlMax=100
        wlInitLow = 9
        wlInitHigh = 11
        wlInit1=random.uniform(wlInitLow,wlInitHigh)
        wlInit2=random.uniform(wlInitLow,wlInitHigh)
                
        ctrlThreshLower = 20
        ctrlThreshUpper = 80
        
        numSteps = 50
        numStepsPRISM = 10
        contAction1 = 0
        contAction2 = 0
        
        unsafe = 0

        mu = 0
        sigma = 5
        # noiseDist = makedist('Normal','mu',mu,'sigma',sigma)
        np.random.normal(mu,sigma,1)[0] ## SEEME: command to generate samples from noise distribution
        # noiseDist = scipy.stats.norm(mu,sigma)
        minValProb = 0.1
        maxValProb = 0.1


        estimated_safety_probs = []
        true_safety_probs = []
        safe_unsafe_over_time = []


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


        for i in range(numSteps):
            r=random.uniform(0,1)
            if(r<minValProb):
                noise1=-100
            elif(r>(1-maxValProb)):
                noise1=100
            else:
                noise1 = np.random.normal(mu,sigma,1)[0]
            
            noises1.append(noise1)
            wlPer1 = max(min(wl1+noise1,wlMax),0)
            wlPers1.append(wlPer1)
            
            stateDist1 = bayesMonitorPerception(stateDist1,wlPer1,mu,sigma,filter_wl_disc,minValProb,maxValProb,wlMax)
            wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
            
            
            r=random.uniform(0,1)
            if(r<minValProb):
                noise2=-100
            elif(r>(1-maxValProb)):
                noise2=100
            else:
                noise2 = np.random.normal(mu,sigma,1)[0]
            
            noises2.append(noise2)
            wlPer2 = max(min(wl2+noise2,wlMax),0)
            wlPers2.append(wlPer2)
            
            stateDist2 = bayesMonitorPerception(stateDist2,wlPer2,mu,sigma,filter_wl_disc,minValProb,maxValProb,wlMax)
            wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)

            ## compute control tank 1
            if wlEst1<ctrlThreshLower or (wlEst1<ctrlThreshUpper and contAction1==1):
                contAction1=1
            else:
                contAction1=0
            
            
            ## compute control tank 2
            if wlEst2<ctrlThreshLower or (wlEst1<ctrlThreshUpper and contAction2==1):
                contAction2=1
            else:
                contAction2=0
            

            
            
            ## Global controller
            contActionG1=contAction1
            contActionG2=contAction2
            if(contAction1==1 and contAction2==1 and wlPer1<wlPer2):
                contActionG1=1
                contActionG2=0
            elif(contAction1==1 and contAction2==1 and wlPer1>wlPer2):
                contActionG1=0
                contActionG2=1
            elif(contAction1==1 and contAction2==1 and wlPer1 == wlPer2):
                r = random.uniform(0,1)
                if r<=0.5:
                    contActionG1=1
                    contActionG2=0
                else:
                    contActionG1=0
                    contActionG2=1
            
            
            
            wl1=wl1-random.choice(outflows)+contActionG1*random.choice(inflows)
            
            wls1.append(wl1)

            wl2=wl2-random.choice(outflows)+contActionG2*random.choice(inflows)
            
            wls2.append(wl2)

            ## update filter
            stateDist1 = bayesMonitorDynamics(stateDist1,contActionG1,inflows_est,outflows_est,filter_wl_disc,wlMax)
            wlEst1 = wlEstFromStateDist(stateDist1,filter_wl_disc)
            wlEsts1.append(wlEst1)
            wlEstErrs1.append(wlEst1-wl1)
            allPerErrs.append(wlEst1-wl1)

            stateDist2 = bayesMonitorDynamics(stateDist2,contActionG2,inflows_est,outflows_est,filter_wl_disc,wlMax)
            wlEst2 = wlEstFromStateDist(stateDist2,filter_wl_disc)
            wlEsts2.append(wlEstFromStateDist)
            wlEstErrs2.append(wlEst2-wl2)
            allPerErrs.append(wlEst2-wl2)

            ## ADDME: compute water tank safety here
            wlid1 = math.ceil(max(min(wlEst1,wlMax),0)/delta_wl)
            wlid2 = math.ceil(max(min(wlEst2,wlMax),0)/delta_wl)
            tankSafeProbThisTime = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir)

            ## ADDME: compute water tank safety here
            wlid1 = math.ceil(max(min(wl1,wlMax),0)/delta_wl)
            wlid2 = math.ceil(max(min(wl2,wlMax),0)/delta_wl)
            if wl1 <= 0 or wl2 <= 0 or wl1>wlMax or wl2>wlMax:
                tankTrueSafeProbThisTime=0
            else:
                tankTrueSafeProbThisTime = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,safe_prob_base_dir)

            estimated_safety_probs.append(tankSafeProbThisTime)
            true_safety_probs.append(tankTrueSafeProbThisTime)

            if(wl1<=0 or wl1>wlMax):
                unsafe=1
                safe_unsafe_over_time.append(1)
                break
            elif(wl2<=0 or wl2>wlMax):
                unsafe=1
                safe_unsafe_over_time.append(1)
                break
            else:
                safe_unsafe_over_time.append(0)

            # print("Safety prob: " + str(tankSafeProbThisTime))
            # input("Wait")
        
        allTrialLengths.append(len(estimated_safety_probs))

        unsafes=unsafes+unsafe
        
        for time,safeProb in enumerate(estimated_safety_probs[0:numSteps-numStepsPRISM]):
            allSafeProbs.append(safeProb)
            allTrueSafeProbs.append(true_safety_probs[time])

            if sum(safe_unsafe_over_time[time:time+numStepsPRISM]) >= 1:
                crash_this_time = 1
            else:
                crash_this_time = 0
            allSafeUnsafe.append(crash_this_time)
            if time in safe_probs_per_time:
                safe_probs_per_time[time].append(safeProb)
                safe_unsafe_per_time[time].append(crash_this_time)
                true_safe_probs_per_time[time].append(true_safety_probs[time])
            else:
                safe_probs_per_time[time] = [safeProb]
                safe_unsafe_per_time[time] = [crash_this_time]
                true_safe_probs_per_time[time] = [true_safety_probs[time]]

    
    print("Num unsafes: " + str(unsafes))
    print("Prop unsafes: " + str(unsafes/numTrials))


    plt.hist(allPerErrs,bins=10)
    plt.savefig("testTankSimPython.png")
    plt.clf()

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

    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts = {}
    binned_counts_true = {}
    for bin in range(num_bins):
        binned_counts[bin] = [0,0]
        binned_counts_true[bin] = [0,0]


    for i in range(len(allSafeProbs)):
        conf = allSafeProbs[i]
        confTrue = allTrueSafeProbs[i]
        safeUnsafe = allSafeUnsafe[i] # 0 if safe, 1 if unsafe
        
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts[bin][0]+=safeUnsafe
        binned_counts[bin][1]+=1

        bin = get_bin(confTrue,num_bins)
        binned_counts_true[bin][0]+=safeUnsafe
        binned_counts_true[bin][1]+=1

    if plotMonitorCalibration:
        print("Calibration across all classes conf monitor using state estimates")
        for bin in binned_counts:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            if binned_counts[bin][1] != 0:
                # print(binned_counts[bin])
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts[bin][0]/binned_counts[bin][1])) + ", " + str(binned_counts[bin][1]-binned_counts[bin][0]) + "/" + str(binned_counts[bin][1]))
            else:
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")

        print("Calibration across all classes conf monitor using state estimates")
        for bin in binned_counts_true:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            if binned_counts_true[bin][1] != 0:
                # print(binned_counts[bin])
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts_true[bin][0]/binned_counts_true[bin][1])) + ", " + str(binned_counts_true[bin][1]-binned_counts_true[bin][0]) + "/" + str(binned_counts_true[bin][1]))
            else:
                print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")


        print("Plotting safety vals")
        x_vals_inst = []
        y_vals_inst = []

        for bin in binned_counts:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            bin_avg = (bin_upper+bin_lower)/2

            if binned_counts[bin][1] != 0:
                x_vals_inst.append(bin_avg)
                y_vals_inst.append(1-float(binned_counts[bin][0]/binned_counts[bin][1]))
        
        plt.plot(x_vals_inst,y_vals_inst, 'r-')
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.legend(["Instantaneous Classifier Confidence", "Average Classifier Confidence"])
        plt.savefig(plotSaveDir + "/monitorOutputVEmpiricalSafety.png")



    ## Compute Brier score
    y_true = [1-x for x in allSafeUnsafe] # ground truth labels
    y_probas = allSafeProbs # predicted probabilities generated by sklearn classifier
    y_probs_GT_state = allTrueSafeProbs
    brier_loss = brier_score_loss(y_true, y_probas)
    brier_loss_GT_state = brier_score_loss(y_true, y_probs_GT_state)
    print("Brier Score: " + str(brier_loss))
    print("Brier Score GT State: " + str(brier_loss_GT_state))

    if plotAUCCurve:

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        roc_auc = auc(fpr, tpr)

        fpr_GT_state, tpr_GT_state, thresholds_GT_state = roc_curve(y_true, y_probs_GT_state)
        roc_auc_GT_state = auc(fpr_GT_state, tpr_GT_state)


        # Plot ROC curve
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
