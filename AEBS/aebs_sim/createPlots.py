from car_with_AEBS import World
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import math
import pickle

from car_monitor_particle_filter import particleFilter

from parsePRISMOutput import computeViolationAEBSKnownControl

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import brier_score_loss
import pandas as pd



GOAL_CAR_SPEED = 10 # m/s
CAR_VEL_LOWER_BOUND_CAR = 6 # m/s #FIXME: was 4
CAR_VEL_UPPER_BOUND_CAR = 12 # m/s #FIXME: was 10
CAR_POS_FOLLOWING_LOWER_BOUND = 15 # m
CAR_POS_FOLLOWING_UPPER_BOUND = 25#30 # m
CAR_VEL_LOWER_BOUND_NOTHING = 1.5 # m/s #FIXME: was 2
CAR_VEL_UPPER_BOUND_NOTHING = 1.5 # m/s #FIXME: was 2







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



def main():

    random.seed(346457)


    dataSaveDir = "../results/AEBS_sim/run1_bugFixCrashVar_carObst_noObstDet_distDisc0.25/"
    plotSaveDir = dataSaveDir + "plots/"
    os.makedirs(plotSaveDir,exist_ok=True)
    trialSaveDir = plotSaveDir + "trials/"
    os.makedirs(trialSaveDir,exist_ok=True)
    trialSafeSaveDirSafe = plotSaveDir + "trialsSafe/"
    os.makedirs(trialSafeSaveDirSafe,exist_ok=True)
    trialSafeSaveDirUnsafe = plotSaveDir + "trialsUnsafe/"
    os.makedirs(trialSafeSaveDirUnsafe,exist_ok=True)
    plotDataDir = dataSaveDir + "/data"
    os.makedirs(plotDataDir,exist_ok=True)


    allSafeProbs = []
    allSafeProbsStateDist = []
    allTrueSafeProbs = []
    allSafeUnsafe = []



    plotAUCCurve = True
    plotMonitorCalibration = True



    with open(plotDataDir + "allSafeProbs.pkl",'rb') as f:
        allSafeProbs = pickle.load(f)
    with open(plotDataDir + "allSafeProbsStateDist.pkl",'rb') as f:
        allSafeProbsStateDist = pickle.load(f)
    with open(plotDataDir + "allTrueSafeProbs.pkl",'rb') as f:
        allTrueSafeProbs = pickle.load(f)
    with open(plotDataDir + "allSafeUnsafe.pkl",'rb') as f:
        allSafeUnsafe = pickle.load(f)
    

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

    ## compute ECE and ECCE
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
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(plotSaveDir + "/safetyMonitorAUCCurve_postProcessing.png")







if __name__ == '__main__':
    main()
