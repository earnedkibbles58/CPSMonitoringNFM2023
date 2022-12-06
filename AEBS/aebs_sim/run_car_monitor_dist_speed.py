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




def computeViolationFromDistanceSpeedObst(d,v,obst_type_str,obst_type_int,dist_disc,vel_disc,accs):

    if obst_type_int == 0:
        folder_name = "../../../results/AEBS/violationPolys/vel_" + str(int(v/vel_disc)) + "/"
    else:
        folder_name = "../../../results/AEBS/violationPolys/dist_" + str(int(d/dist_disc)) + "_vel_" + str(int(v/vel_disc)) + "/"
    file_name  = folder_name + "violationPoly" + str(obst_type_str.capitalize()) + ".txt"

    print(file_name)
    print(obst_type_int)
    with open(file_name, "r") as f:
        count = 0
        for line in f:
            count += 1
            if not "Result" in line:
                continue
            # print("Line " + str(count))
            # print(line)

            line = line.strip()
            line = line.split("{")
            # print(line)
            temp_line = line[-1].split("}")[0]
            # print("Temp line")
            # print(temp_line)

            temp_line = temp_line.replace(" * ", "*")
            temp_line = temp_line.replace(" + ", "+")
            temp_line = temp_line.replace(" - ", "-")
            temp_line = temp_line.strip()
            # print(temp_line)

            if obst_type_int == 0:
                temp_line = temp_line.replace(" Pnn", "*Pnn")
                temp_line = temp_line.replace(" Prn", "*Prn")
                temp_line = temp_line.replace("^", "**")
                # temp_line = temp_line.replace(" Pcn")
                
                temp_line = temp_line.replace("Pnn", str(accs[obst_type_int][0]))
                temp_line = temp_line.replace("Prn", str(accs[obst_type_int][1]))
            elif obst_type_int == 1:
                temp_line = temp_line.replace(" Pnr", "*Pnr")
                temp_line = temp_line.replace(" Prr", "*Prr")
                temp_line = temp_line.replace("^", "**")
                # temp_line = temp_line.replace(" Pcn")
                
                temp_line = temp_line.replace("Pnr", str(accs[obst_type_int][0]))
                temp_line = temp_line.replace("Prr", str(accs[obst_type_int][1]))
            elif obst_type_int == 2:
                print("Replacing car")
                temp_line = temp_line.replace(" Pnc", "*Pnc")
                temp_line = temp_line.replace(" Prc", "*Prc")
                temp_line = temp_line.replace("^", "**")
                # temp_line = temp_line.replace(" Pcn")
                
                temp_line = temp_line.replace("Pnc", str(accs[obst_type_int][0]))
                temp_line = temp_line.replace("Prc", str(accs[obst_type_int][1]))
            else:
                raise Exception("Incorrect obst int " + str(obst_type_int))


            # print("Temp line")
            # print(temp_line)

            try:
                violation_prob = eval(temp_line)
                print("Violation prob")
                print(violation_prob)

                return violation_prob
            except Exception as e:
                print("Error in evaluation")
                return -1
    
    print("Couldn't find result for " + file_name)
    return -1

            


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



def compute_obst_dist(accs, det_counts):
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

    return obst_probs

def compute_safety_prob_fixed_acc(accs, det_counts):


    obst_probs = compute_obst_dist(accs, det_counts)

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
    plotEachTrial = True
    plotAUCCurve = True

    episode_length = 30
    time_step = 0.1

    init_car_dist = 25
    init_car_vel = 8
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

    safe_probs_per_time = {}
    safe_unsafe_per_time = {}

    plotSaveDir = "../results/monitorPlots/distSpeedMonitor/nothing/"
    trialSaveDir = plotSaveDir + "trials/"

    useObstPredictions = False

    for step in range(numTrajectories):

        print("Trial " + str(step))

        # true_obst = random.choice([0,1,2])
        true_obst = 0

        true_obst_str = obsts_str[true_obst]
        true_obst_vel = obsts_vel[true_obst]

        w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

        dist_speed_violation_prob_prev = 0.5
        temp_violation_prob_prev = [0.5,0.5,0.5]

        all_preds = []
        tempSafeProbs = []
        pred_counts = [0,0,0]

        all_confs = []
        dist_speed_safety_probs = []

        car_dist = init_car_dist
        car_vel = init_car_vel


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
                obj_dist_mon = 15
                obj_vel = 0
            elif pred_obj == "rock":
                if true_obst_str == "nothing":
                    obj_dist = ERRONEOUS_OBJ_DET_DIST
                    obj_dist_mon = obj_dist
                else:
                    obj_dist = w.car_dist
                    obj_dist_mon = obj_dist
                obj_vel = 0
            elif pred_obj == "car":
                if true_obst_str == "nothing":
                    obj_dist = ERRONEOUS_OBJ_DET_DIST
                    obj_dist_mon = obj_dist
                else:
                    obj_dist = w.car_dist
                    obj_dist_mon = obj_dist
                obj_vel = obsts_vel[2]
            # print(obj_vel)

            if useObstPredictions:
                violation_prob = 0
                for pred_obst in [0,1,2]:
                    temp_violation_prob = computeViolationFromDistanceSpeedObst(obj_dist_mon,car_vel,obsts_str[pred_obst],pred_obst,dist_disc,vel_disc,fixed_accs)
                    if temp_violation_prob == -1:
                        temp_violation_prob = temp_violation_prob_prev[pred_obst]
                    temp_violation_prob_prev[pred_obst] = temp_violation_prob
                    violation_prob+= obst_probs[pred_obst]*temp_violation_prob
                dist_speed_safety_probs.append(1-violation_prob)
            else:
                if true_obst == 0:
                    dist_speed_violation_prob = computeViolationFromDistanceSpeedObst(init_car_dist,car_vel,true_obst_str,true_obst,dist_disc,vel_disc,fixed_accs)
                else:
                    dist_speed_violation_prob = computeViolationFromDistanceSpeedObst(w.car_dist,car_vel,true_obst_str,true_obst,dist_disc,vel_disc,fixed_accs)
                if dist_speed_violation_prob == -1:
                    dist_speed_violation_prob = dist_speed_violation_prob_prev
                    print("Using prev violation prob")
                dist_speed_violation_prob_prev = dist_speed_violation_prob
                dist_speed_safety_probs.append(1-dist_speed_violation_prob)



            car_dist, car_vel, crash, done = w.step(pred_obj, obj_dist, obj_vel)

            if done:
                if crash:
                    num_unsafe += 1
                break
            
        
        print("Safety probs")
        print(dist_speed_safety_probs)
        
        allTrialLengths.append(len(dist_speed_safety_probs))
        if crash:
            for time,safeProb in enumerate(dist_speed_safety_probs):
                allSafeProbs.append(safeProb)
                allSafeUnsafe.append(1)
                if time in safe_probs_per_time:
                    safe_probs_per_time[time].append(safeProb)
                    safe_unsafe_per_time[time].append(1)
                else:
                    safe_probs_per_time[time] = [safeProb]
                    safe_unsafe_per_time[time] = [1]

        else:
            for time,safeProb in enumerate(dist_speed_safety_probs):
                allSafeProbs.append(safeProb)
                allSafeUnsafe.append(0)
                if time in safe_probs_per_time:
                    safe_probs_per_time[time].append(safeProb)
                    safe_unsafe_per_time[time].append(0)
                else:
                    safe_probs_per_time[time] = [safeProb]
                    safe_unsafe_per_time[time] = [0]

        
        if plotEachTrial:
            os.makedirs(trialSaveDir,exist_ok=True)
            # plt.clf()
            # plt.plot(tempSafeProbs,'k-')
            # plt.ylim(0,1)
            # plt.xlabel("time")
            # plt.ylabel("Safety Probability")
            # if crash:
            #     plt.title("Obstacle: " + str(true_obst_str) + ", crash")
            # else:
            #     plt.title("Obstacle: " + str(true_obst_str) + ", safe")
            # plt.savefig(trialSaveDir + "trial%04d.png" % step)
            # plt.clf()

            plt.plot(dist_speed_safety_probs,'k-')
            plt.ylim(-0.1,1.1)
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


    print("Calibration for obstacle predictor: Car")
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
        
        plt.plot(x_vals_inst,y_vals_inst, 'r-')
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.legend(["Instantaneous Classifier Confidence", "Average Classifier Confidence"])
        plt.savefig(plotSaveDir + "/monitorOutputVEmpiricalSafety.png")



    ## Compute Brier score
    y_true = [1-x for x in allSafeUnsafe] # ground truth labels
    y_probas = allSafeProbs # predicted probabilities generated by sklearn classifier
    brier_loss = brier_score_loss(y_true, y_probas)
    print("Brier Score: " + str(brier_loss))

    print("Safety probs monitor " + str(allSafeProbs))

    if plotAUCCurve:

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
        plt.savefig(plotSaveDir + "/safetyMonitorAUCCurve.png")



        ## compute scores over time
        brier_scores_over_time = []
        for e in range(episode_length):
            safe_probs = safe_probs_per_time[e]
            safe_unsafe = safe_unsafe_per_time[e]

            ## Compute Brier score
            y_true = [1-x for x in safe_unsafe] # ground truth labels
            y_probas = safe_probs # predicted probabilities generated by sklearn classifier
            brier_loss = brier_score_loss(y_true, y_probas)
            print("Brier Score: " + str(brier_loss))
            brier_scores_over_time.append(brier_loss)

        plt.clf()
        plt.plot(brier_scores_over_time)
        plt.xlabel("Time step")
        plt.ylabel("Brier loss")
        plt.title("Brier score over time")
        plt.gca().set_ylim(bottom=0)
        plt.gca().set_ylim(top=1)
        plt.savefig(plotSaveDir + "/brierScoresOverTime.png")

        ## auc score over time
        auc_over_time = []
        for e in range(episode_length):
            safe_probs = safe_probs_per_time[e]
            safe_unsafe = safe_unsafe_per_time[e]

            y_true = [1-x for x in safe_unsafe] # ground truth labels
            y_probas = safe_probs # predicted probabilities generated by sklearn classifier

            # Compute fpr, tpr, thresholds and roc auc
            fpr, tpr, thresholds = roc_curve(y_true, y_probas)
            roc_auc = auc(fpr, tpr)
            auc_over_time.append(roc_auc)

            # Plot ROC curve
            plt.clf()
            plt.plot(fpr, tpr, label="ROC curve (area = roc_auc) time " + str(e))
            plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate or (1 - Specifity)')
            plt.ylabel('True Positive Rate or (Sensitivity)')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(plotSaveDir + "/AUCCurveTime" + str(e) + ".png")
        
        print(auc_over_time)
        plt.clf()
        plt.plot(auc_over_time)
        plt.title("AUC score over time")
        plt.xlabel("Time step")
        plt.ylabel("AUC")
        plt.savefig(plotSaveDir + "/AUCScoreOverTime.png")



if __name__ == '__main__':
    main()
