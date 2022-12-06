from asyncio import ALL_COMPLETED
from curses.ascii import FF
from inspect import trace
from site import makepath
from car_with_AEBS import World
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import math


from testCalibratedModel import calibrated_perception
from runtime_monitor import prismViolationProbNothing, prismViolationProbRock, prismViolationProbCar

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import brier_score_loss

def classification_model_fixed_acc(true_obst,accs):

    # accs: 3x3 array of accuracy of each class. accs[0][2] is the probability of the preception returning car if there is nothing in the environment

    confs = [1/3,1/3,1/3]

    det_probs = accs[true_obst]
    labels = [0,1,2]
    det = random.choices(labels,weights = det_probs, k=1)[0]

    return det, confs




def compute_acc_given_gt(preds,gt_label):

    total_pred = len(preds)
    correct_pred = 0

    for pred in preds:
        if pred == gt_label:
            correct_pred += 1
    
    return correct_pred/total_pred



def compute_safety_prob(conditional_accs, confs):
    safety_prob_nothing = 1-prismViolationProbNothing(conditional_accs[0],conditional_accs[1])
    safety_prob_rock = 1-prismViolationProbRock(conditional_accs[0],conditional_accs[1])
    safety_prob_car = 1-prismViolationProbCar(conditional_accs[0],conditional_accs[1])

    safety_prob = confs[0]*safety_prob_nothing + confs[1]*safety_prob_rock + confs[2]*safety_prob_car

    return safety_prob

def get_bin(conf,num_bins):

    # print(conf)
    for j in range(num_bins):
        # print((j+1)*(1/num_bins))
        if conf<(j+1)*(1/num_bins):
            return j
    return num_bins-1


def compute_safety_prob_fixed_acc(accs, det_counts):


    labels = [0,1,2]

    obst_probs = []
    for label in labels:
        accs_this_obst = accs[label]
        p_obst = 1
        for i in range(len(accs_this_obst)):
            p_obst *= accs_this_obst[i]**det_counts[i]
        obst_probs.append(p_obst)
    
    normalize_sum = sum(obst_probs)
    for i in range(len(obst_probs)):
        obst_probs[i] = obst_probs[i]/normalize_sum
    
    print("Obstacle probs: " + str(obst_probs))
    assert abs(sum(obst_probs)-1)<=0.0001

    safety_prob_nothing = 1-prismViolationProbNothing(accs[0][0],accs[0][1])
    safety_prob_rock = 1-prismViolationProbRock(accs[1][0],accs[1][1])
    safety_prob_car = 1-prismViolationProbCar(accs[2][0],accs[2][1])

    safe_prob = obst_probs[0]*safety_prob_nothing + obst_probs[1]*safety_prob_rock + obst_probs[2]*safety_prob_car
    return safe_prob, obst_probs


def compute_PRISM_safety_fixed_accs(accs):
    safety_prob_nothing = 1-prismViolationProbNothing(accs[0][0],accs[0][1])
    safety_prob_rock = 1-prismViolationProbRock(accs[1][0],accs[1][1])
    safety_prob_car = 1-prismViolationProbCar(accs[2][0],accs[2][1])

    print("Safe prob nothing: " + str(safety_prob_nothing))
    print("Safe prob rock: " + str(safety_prob_rock))
    print("Safe prob car: " + str(safety_prob_car))

def main(initSeeds = None,initNumTrajectories=100000):

    if initSeeds is None:
        numTrajectories = initNumTrajectories
    else:
        numTrajectories = len(initSeeds)

    numTrajectories = 100
    plotMonitorCalibration = False
    plotEachTrial = False
    plotAUCCurve = True

    episode_length = 30
    time_step = 0.1

    init_car_dist = 30
    init_car_vel = 10
    num_unsafe = 0

    dist_disc = 0.5
    vel_disc = 0.4

    ERRONEOUS_OBJ_DET_DIST = 10

    w = World(init_car_dist, init_car_vel, "nothing", 0, dist_disc, vel_disc, episode_length, time_step)

    obsts_str = ["nothing","rock","car"]
    obsts_vel = [0,0,8]
    
    allDist = []
    allVel = []

    fixed_accs = [[0.5,0.25,0.25],[0.4,0.2,0.4],[0.2,0.5,0.3]]

    compute_PRISM_safety_fixed_accs(fixed_accs)

    allSafeProbs = []
    allSafeProbsAvgConf = []
    allSafeUnsafe = []
    
    allTrialLengths = []

    allObstProbs = []
    allObsts = []

    for step in range(numTrajectories):

        true_obst = random.choice([0,1,2])
        # true_obst = 2

        true_obst_str = obsts_str[true_obst]
        true_obst_vel = obsts_vel[true_obst]

        w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

        all_preds = []
        tempSafeProbs = []
        pred_counts = [0,0,0]

        all_confs = []
        ## keep track of previous 
        for e in range(episode_length):


            pred_obj_ind, confs = classification_model_fixed_acc(true_obst,fixed_accs)
            pred_counts[pred_obj_ind]+=1

            pred_obj = obsts_str[pred_obj_ind]

            all_preds.append(pred_obj)
            all_confs.append(confs)

            safety_prob,obst_probs = compute_safety_prob_fixed_acc(fixed_accs,pred_counts)
            tempSafeProbs.append(safety_prob)

            allObstProbs.append(obst_probs)
            allObsts.append(true_obst)


            if pred_obj == "nothing":
                obj_dist = 0
                obj_vel = 0
            elif pred_obj == "rock":
                if true_obst_str == "nothing":
                    obj_dist = ERRONEOUS_OBJ_DET_DIST
                else:
                    obj_dist = w.car_dist
                obj_vel = 0
            elif pred_obj == "car":
                if true_obst_str == "nothing":
                    obj_dist = ERRONEOUS_OBJ_DET_DIST
                else:
                    obj_dist = w.car_dist
                obj_vel = obsts_vel[2]
            # print(obj_vel)
            car_dist, car_vel, crash, done = w.step(pred_obj, obj_dist, obj_vel)



            if done:
                if crash:
                    num_unsafe += 1
                break
        
        allTrialLengths.append(len(tempSafeProbs))
        if crash:
            for safeProb in tempSafeProbs:
                allSafeProbs.append(safeProb)
                allSafeUnsafe.append(1)
        else:
            for safeProb in tempSafeProbs:
                allSafeProbs.append(safeProb)
                allSafeUnsafe.append(0)

        trialSaveDir = "../results/monitorPlots/fixedAccStateEst/trials/"
        if plotEachTrial:
            plt.clf()
            plt.plot(tempSafeProbs,'k-')
            plt.ylim(0,1)
            plt.xlabel("time")
            plt.ylabel("Safety Probability")
            if crash:
                plt.title("Obstacle: " + str(true_obst_str) + ", crash")
            else:
                plt.title("Obstacle: " + str(true_obst_str) + ", safe")
            plt.savefig(trialSaveDir + "trial%04d.png" % step)
            plt.clf()

        allDist.append(w.allDist)
        allVel.append(w.allVel)


    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts = {}
    for bin in range(num_bins):
        binned_counts[bin] = [0,0]


    for i in range(len(allSafeProbs)):
        conf = allSafeProbs[i]
        safeUnsafe = allSafeUnsafe[i] # 0 if safe, 1 if unsafe
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts[bin][0]+=safeUnsafe
        binned_counts[bin][1]+=1


    print("Calibration across all classes for instantaneous conf monitor")
    for bin in binned_counts:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts[bin][0]/binned_counts[bin][1])) + ", " + str(binned_counts[bin][1]-binned_counts[bin][0]) + "/" + str(binned_counts[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")



    ## check for calibration for none
    # bin at level of 0.1
    num_bins = 10
    binned_counts_none = {}
    for bin in range(num_bins):
        binned_counts_none[bin] = [0,0]

    for i in range(len(allObsts)):
        conf = allObstProbs[i][0]
        safeUnsafe = 1 if allObsts[i]==0 else 0 # 1 correct, 0 if uncorrect
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts_none[bin][0]+=safeUnsafe
        binned_counts_none[bin][1]+=1


    print("Calibration for obstacle predictor: None")
    for bin in binned_counts_none:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts_none[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(float(binned_counts_none[bin][0]/binned_counts_none[bin][1])) + ", " + str(binned_counts_none[bin][0]) + "/" + str(binned_counts_none[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")


    ## check for calibration rock
    # bin at level of 0.1
    num_bins = 10
    binned_counts_rock = {}
    for bin in range(num_bins):
        binned_counts_rock[bin] = [0,0]

    for i in range(len(allObsts)):
        conf = allObstProbs[i][1]
        safeUnsafe = 1 if allObsts[i]==1 else 0 # 1 correct, 0 if uncorrect
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts_rock[bin][0]+=safeUnsafe
        binned_counts_rock[bin][1]+=1


    print("Calibration for obstacle predictor: Rock")
    for bin in binned_counts_rock:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts_rock[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(float(binned_counts_rock[bin][0]/binned_counts_rock[bin][1])) + ", " + str(binned_counts_rock[bin][0]) + "/" + str(binned_counts_rock[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")



    ## check for calibration car
    # bin at level of 0.1
    num_bins = 10
    binned_counts_car = {}
    for bin in range(num_bins):
        binned_counts_car[bin] = [0,0]

    for i in range(len(allObsts)):
        conf = allObstProbs[i][2]
        safeUnsafe = 1 if allObsts[i]==2 else 0 # 1 correct, 0 if uncorrect
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts_car[bin][0]+=safeUnsafe
        binned_counts_car[bin][1]+=1


    print("Calibration for obstacle predictor: Rock")
    for bin in binned_counts_car:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts_car[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(float(binned_counts_car[bin][0]/binned_counts_car[bin][1])) + ", " + str(binned_counts_car[bin][0]) + "/" + str(binned_counts_car[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")

    
    print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))
    print('average length of scenario: ' + str(np.mean(allTrialLengths)))
    if plotMonitorCalibration:
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
        
        trialSaveDir = "../results/monitorPlots/fixedAccStateEst/"
        plt.plot(x_vals_inst,y_vals_inst, 'r-')
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.legend(["Instantaneous Classifier Confidence", "Average Classifier Confidence"])
        plt.savefig(trialSaveDir + "/monitorOutputVEmpiricalSafety.png")



    ## Comptue Brier score
    y_true = [1-x for x in allSafeUnsafe] # ground truth labels
    y_probas = allSafeProbs # predicted probabilities generated by sklearn classifier
    brier_loss = brier_score_loss(y_true, y_probas)
    print("Brier Score: " + str(brier_loss))

    if plotAUCCurve:
        trialSaveDir = "../results/monitorPlots/fixedAccStateEst/"

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(trialSaveDir + "/safetyMonitorAUCCurve.png")

if __name__ == '__main__':
    main()
