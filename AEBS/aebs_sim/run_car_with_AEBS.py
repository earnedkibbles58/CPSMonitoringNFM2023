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


def classification_model(true_obst):

    # return "nothing"
    # return "rock"
    # return "car"

    confs = [1/3,1/3,1/3]
    randNum = random.random()
    if randNum <= 1/3:
        return "nothing", confs
    elif randNum <= 2/3:
        return "rock", confs
    else:
        return "car", confs

    return random.choice(["nothing","rock","car"])



def classification_model_tv(true_obst,t,accs,times):

    # if t <= 15:
    #     correct_prob = 0.3
    # else:
    #     correct_prob = 0.5
    
    assert len(accs) == len(times)
    # print(accs)
    # print(times)
    # print(t)
    
    correct_prob = accs[0]
    if t >= max(times):
        correct_prob = accs[-1]
    else:
        for i in range(len(accs)):
            acc = accs[i]
            time = times[i]
            if t > time:
                break

            correct_prob = acc


    # print("Correct prob: " + str(correct_prob))
    confs = [1/3,1/3,1/3]
    labels = ["nothing","rock","car"]
    randNum = random.random()
    if randNum <= correct_prob:
        return true_obst, confs
    else:
        labels.remove(true_obst)
        return random.choice(labels), confs


def calibrated_perception_model(true_obst, obst_strs):
    true_obst_ind = obst_strs.index(true_obst)

    sorted_confs = calibrated_perception(true_obst_ind)

    return obst_strs[sorted_confs.index(max(sorted_confs))], sorted_confs


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


    allSafeProbs = []
    allSafeProbsAvgConf = []
    allSafeUnsafe = []
    
    allTrialLengths = []

    for step in range(numTrajectories):

        true_obst = random.choice([0,1,2])
        # true_obst = 2

        true_obst_str = obsts_str[true_obst]
        true_obst_vel = obsts_vel[true_obst]

        w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

        all_preds = []
        tempSafeProbs = []
        tempSafeProbsAverageConf = []

        all_confs = []
        ## keep track of previous 
        for e in range(episode_length):

            # pred_obj, confs = classification_model(true_obst_str)
            # pred_obj, confs = classification_model_tv(true_obst_str,e)
            pred_obj, confs = calibrated_perception_model(true_obst_str, obsts_str)
            all_preds.append(pred_obj)
            all_confs.append(confs)

            conditional_accs = []
            for gt_label in obsts_str:
                conditional_accs.append(compute_acc_given_gt(all_preds,gt_label))
            
            # print(true_obst_str)
            # print(all_preds)
            # print(conditional_accs)
            # input("WAIT")

            average_confs = np.mean(all_confs,0)
            # print(sum(average_confs))
            assert abs(sum(average_confs)-1) <= 0.0001
            safe_prob = compute_safety_prob(conditional_accs, confs)
            tempSafeProbs.append(safe_prob)

            safe_prob_avg = compute_safety_prob(conditional_accs, average_confs)
            tempSafeProbsAverageConf.append(safe_prob_avg)

            # print("Safety: " + str(safe_prob))
            # input("WAIT")

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
                #     for safe in tempSafeProbs:
                #         allSafeProbs.append(safe)
                #         allSafeUnsafe.append(1)
                # else:
                #     for safe in tempSafeProbs:
                #         allSafeProbs.append(safe)
                #         allSafeUnsafe.append(0)
                break
        
        allTrialLengths.append(len(tempSafeProbs))
        if crash:
            trialSaveDir = "../results/monitorPlots/indivTrials_varyingConf/nothing/crash/"
            for safeProb in tempSafeProbs:
                allSafeProbs.append(safeProb)
                allSafeUnsafe.append(1)
            for safeProb in tempSafeProbsAverageConf:
                allSafeProbsAvgConf.append(safeProb)
        else:
            trialSaveDir = "../results/monitorPlots/indivTrials_varyingConf/nothing/safe/"
            for safeProb in tempSafeProbs:
                allSafeProbs.append(safeProb)
                allSafeUnsafe.append(0)
            for safeProb in tempSafeProbsAverageConf:
                allSafeProbsAvgConf.append(safeProb)

        if plotEachTrial:
            plt.clf()
            plt.plot(tempSafeProbs,'k-')
            plt.plot(tempSafeProbsAverageConf,'b-')
            plt.ylim(0,1)
            plt.legend(["Instantaneous Classifier Confidence", "Average Classifier Confidence"])
            plt.xlabel("time")
            plt.ylabel("Safety Probability")
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







    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts_avg = {}
    for bin in range(num_bins):
        binned_counts_avg[bin] = [0,0]


    for i in range(len(allSafeProbsAvgConf)):
        conf = allSafeProbsAvgConf[i]
        safeUnsafe = allSafeUnsafe[i] # 0 if safe, 1 if unsafe
        
        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts_avg[bin][0]+=safeUnsafe
        binned_counts_avg[bin][1]+=1


    print("Calibration across all classes for average conf monitor")
    for bin in binned_counts_avg:
        bin_lower = bin/num_bins
        bin_upper = (bin+1)/num_bins
        if binned_counts_avg[bin][1] != 0:
            # print(binned_counts[bin])
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: " + str(1-float(binned_counts_avg[bin][0]/binned_counts_avg[bin][1])) + ", " + str(binned_counts_avg[bin][1]-binned_counts_avg[bin][0]) + "/" + str(binned_counts_avg[bin][1]))
        else:
            print("[" + str(bin_lower) + "," + str(bin_upper) + "]: No data in this bin")




        # plt.plot(dists_to_wall)
        # plt.savefig('distsToWallTrace' + str(step) + '.png')
        # plt.clf()

    
    print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))
    print('average length of scenario: ' + str(np.mean(allTrialLengths)))
    if plotMonitorCalibration:
        print("Plotting safety vals")
        x_vals_inst = []
        x_vals_avg = []
        y_vals_inst = []
        y_vals_avg = []

        for bin in binned_counts:
            bin_lower = bin/num_bins
            bin_upper = (bin+1)/num_bins
            bin_avg = (bin_upper+bin_lower)/2

            if binned_counts[bin][1] != 0:
                x_vals_inst.append(bin_avg)
                y_vals_inst.append(1-float(binned_counts[bin][0]/binned_counts[bin][1]))

            if binned_counts_avg[bin][1] != 0:
                x_vals_avg.append(bin_avg)
                y_vals_avg.append(float(1-binned_counts_avg[bin][0]/binned_counts_avg[bin][1]))

        
        plt.plot(x_vals_inst,y_vals_inst, 'r-')
        plt.plot(x_vals_avg,y_vals_avg, 'b-')
        plt.xlabel("Monitor Safety Probability")
        plt.ylabel("Empirical Safety Probability")
        plt.legend(["Instantaneous Classifier Confidence", "Average Classifier Confidence"])
        plt.savefig("../results/monitorPlots/monitorOutputVEmpiricalSafety_onlyCar_0.3_0.3_0.4.png")




    if plotAUCCurve:
        trialSaveDir = "../results/monitorPlots/indivTrials_varyingConf"
        
        ## compute auc curve
        y_true = [1-x for x in allSafeUnsafe] # ground truth labels
        y_probas = allSafeProbs # predicted probabilities generated by sklearn classifier

        print(min(y_true))
        print(min(y_probas))

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




def main_temporalAccuracyDependenceTest(initSeeds = None,initNumTrajectories=100000):

    if initSeeds is None:
        numTrajectories = initNumTrajectories
    else:
        numTrajectories = len(initSeeds)

    numTrajectories = 10000

    episode_length = 30
    time_step = 0.1

    init_car_dist = 30
    init_car_vel = 10

    dist_disc = 0.5
    vel_disc = 0.4

    ERRONEOUS_OBJ_DET_DIST = 10

    w = World(init_car_dist, init_car_vel, "nothing", 0, dist_disc, vel_disc, episode_length, time_step)

    obsts_str = ["nothing","rock","car"]
    obsts_vel = [0,0,8]
    

    times = [0,episode_length/2]
    acc_bases = [0.4,0.2,0.2]
    acc_deltases = [[0,5,-5,10,-10,-15,15,20,-20,30,-30,40,-40],[0,5,-5,10,-10,-15,15,20,-20],[0,5,-5,10,-10,-15,15,20,-20]]
    # acc_deltas = [0,5,-5,10,-10,-15,15,20,-20]

    true_obsts = [0,1,2]

    for true_obst in true_obsts:
        print("Obstacle: " + obsts_str[true_obst])

        all_crash_probs = []
        acc_base = acc_bases[true_obst]
        acc_deltas = acc_deltases[true_obst]

        for acc_delta in acc_deltas:
            accs = [acc_base+acc_delta/100,acc_base-acc_delta/100]
            print("Accs: " + str(accs))
            
            allTrialLengths = []
            num_unsafe = 0

            for step in range(numTrajectories):

                # true_obst = random.choice([0,1,2])
                # true_obst = 2

                true_obst_str = obsts_str[true_obst]
                true_obst_vel = obsts_vel[true_obst]

                w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

                all_preds = []
                tempSafeProbs = []
                tempSafeProbsAverageConf = []
                car_vels = [init_car_vel]

                all_confs = []
                ## keep track of previous 
                for e in range(episode_length):

                    pred_obj, confs = classification_model_tv(true_obst_str,e,accs,times)
                    all_preds.append(pred_obj)
                    all_confs.append(confs)

                    conditional_accs = []
                    for gt_label in obsts_str:
                        conditional_accs.append(compute_acc_given_gt(all_preds,gt_label))
                    
                    # print(true_obst_str)
                    # print(all_preds)
                    # print(conditional_accs)
                    # input("WAIT")

                    average_confs = np.mean(all_confs,0)
                    # print(sum(average_confs))
                    assert abs(sum(average_confs)-1) <= 0.0001
                    safe_prob = compute_safety_prob(conditional_accs, confs)
                    tempSafeProbs.append(safe_prob)

                    safe_prob_avg = compute_safety_prob(conditional_accs, average_confs)
                    tempSafeProbsAverageConf.append(safe_prob_avg)

                    # print("Safety: " + str(safe_prob))
                    # input("WAIT")

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
                    # print(pred_obj, obj_dist, obj_vel)
                    car_dist, car_vel, crash, done = w.step(pred_obj, obj_dist, obj_vel)
                    car_vels.append(car_vel)



                    if done:
                        if crash:
                            num_unsafe += 1
                            # print(car_vels)
                        break
                
                allTrialLengths.append(len(tempSafeProbs))
                
            all_crash_probs.append(float(num_unsafe/numTrajectories))

            print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))
            print('average length of scenario: ' + str(np.mean(allTrialLengths)),flush=True)

        print(obsts_str[true_obst])
        plt.clf()
        trialSaveDir = "../results/monitorPlots/averageAccExp/"
        plt.plot(acc_deltas,all_crash_probs,'b*')
        plt.xlabel("Delta Classifier Accuracy (First Half of Trace v. Average Accuracy)")
        plt.ylabel("Empirical Crash Probability")
        plt.title("Obstacle: " + obsts_str[true_obst] + ", average acc=" + str(acc_base))
        plt.savefig(trialSaveDir + obsts_str[true_obst] + ".png")





def genPerceptionReadings(gt_label,acc,traceLen):
    ## pick exactly acc*traceLen readings to be correct, rest to be wrong, in random order

    all_labels = [0,1,2]
    all_labels.remove(gt_label)
    wrong_label_1 = random.choice(all_labels)
    all_labels.remove(wrong_label_1)
    wrong_label_2 = all_labels[0]


    num_correct = round(acc*traceLen)
    num_wrong_1 = round((traceLen - num_correct)/2)
    num_wrong_2 = traceLen - num_correct - num_wrong_1

    inds = [i for i in range(traceLen)]

    correct_inds = random.sample(inds,num_correct)

    [inds.remove(i) for i in correct_inds]
    wrong_inds_1 = random.sample(inds,num_wrong_1)
    [inds.remove(i) for i in wrong_inds_1]
    wrong_inds_2 = inds

    assert len(correct_inds) == num_correct and len(wrong_inds_1) == num_wrong_1 and len(wrong_inds_2) == num_wrong_2
    assert len(correct_inds) + len(wrong_inds_1) + len(wrong_inds_2) == traceLen


    pred_labels = []
    for i in range(traceLen):
        if i in correct_inds:
            pred_labels.append(gt_label)
        elif i in wrong_inds_1:
            pred_labels.append(wrong_label_1)
        elif i in wrong_inds_2:
            pred_labels.append(wrong_label_2)
        else:
            raise Exception("Error with label inds in genPerceptionReadings()")

    return pred_labels


def main_exactlyAverageNumRightTest(initSeeds = None,initNumTrajectories=100000):

    if initSeeds is None:
        numTrajectories = initNumTrajectories
    else:
        numTrajectories = len(initSeeds)

    numTrajectories = 10000

    episode_length = 30
    time_step = 0.1

    init_car_dist = 30
    init_car_vel = 10

    dist_disc = 0.5
    vel_disc = 0.4

    ERRONEOUS_OBJ_DET_DIST = 10

    w = World(init_car_dist, init_car_vel, "nothing", 0, dist_disc, vel_disc, episode_length, time_step)

    gt_labels = [0,1,2]
    obsts_str = ["nothing","rock","car"]
    obsts_vel = [0,0,8]
    
    accs = [0.8,0.5,0.4,0.2,0.1]

    for true_obst in gt_labels:
        print("Obst: " + (obsts_str[true_obst]))

        prism_crash_probs = []
        fixed_pred_crash_probs = []

        for acc in accs:
            # acc = accs[true_obst]
            print("Acc: " + str(acc))

            if true_obst == 0:
                prism_crash_prob = prismViolationProbNothing(acc,(1-acc)/2)
            elif true_obst == 1:
                prism_crash_prob = prismViolationProbRock((1-acc)/2,acc)
            elif true_obst == 2:
                prism_crash_prob = prismViolationProbCar((1-acc)/2,(1-acc)/2)
            else:
                raise Exception("Bad obstacle " + str(true_obst))
            print("PRISM crash prob: " + str(prism_crash_prob))
            prism_crash_probs.append(prism_crash_prob)
            allTrialLengths = []
            num_unsafe = 0

            for step in range(numTrajectories):

                true_obst_str = obsts_str[true_obst]
                true_obst_vel = obsts_vel[true_obst]

                w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

                all_preds = []
                tempSafeProbs = []
                tempSafeProbsAverageConf = []
                car_vels = [init_car_vel]

                all_confs = []
                ## keep track of previous
                preds = genPerceptionReadings(true_obst,acc,episode_length)
                # print("Preds: " + str(preds),flush=True)

                for e in range(episode_length):

                    pred_obj = obsts_str[preds[e]]
                    confs = [1/3,1/3,1/3] # doesn't matter here
                    all_preds.append(pred_obj)
                    all_confs.append(confs)

                    conditional_accs = []
                    for gt_label in obsts_str:
                        conditional_accs.append(compute_acc_given_gt(all_preds,gt_label))
                    
                    # print(true_obst_str)
                    # print(all_preds)
                    # print(conditional_accs)
                    # input("WAIT")

                    average_confs = np.mean(all_confs,0)
                    # print(sum(average_confs))
                    assert abs(sum(average_confs)-1) <= 0.0001
                    safe_prob = compute_safety_prob(conditional_accs, confs)
                    tempSafeProbs.append(safe_prob)

                    safe_prob_avg = compute_safety_prob(conditional_accs, average_confs)
                    tempSafeProbsAverageConf.append(safe_prob_avg)

                    # print("Safety: " + str(safe_prob))
                    # input("WAIT")

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
                    else:
                        raise Exception("Issue with pred obj " + str(pred_obj))
                    # print(obj_vel)
                    # print(pred_obj, obj_dist, obj_vel)
                    car_dist, car_vel, crash, done = w.step(pred_obj, obj_dist, obj_vel)
                    car_vels.append(car_vel)



                    if done:
                        if crash:
                            num_unsafe += 1
                            # print(car_vels)
                        break
                
                allTrialLengths.append(len(tempSafeProbs))
                
            fixed_pred_crash_probs.append(float(num_unsafe/numTrajectories))
            print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))
            print('average length of scenario: ' + str(np.mean(allTrialLengths)),flush=True)

        plt.clf()
        trialSaveDir = "../results/monitorPlots/fixedPredExp/"
        plt.plot(accs,prism_crash_probs,'r*')
        plt.plot(accs,fixed_pred_crash_probs,'b*')
        plt.xlabel("Classifier Accuracy")
        plt.ylabel("Empirical Crash Probability")
        plt.legend(["PRISM Crash Prob", "Empirical Crash Prob"])
        plt.savefig(trialSaveDir + obsts_str[true_obst] + ".png")

if __name__ == '__main__':
    main()
    # main_temporalAccuracyDependenceTest()
    # main_exactlyAverageNumRightTest()
