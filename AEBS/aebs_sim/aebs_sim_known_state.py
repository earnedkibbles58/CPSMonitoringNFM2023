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




def classification_model_fixed_acc(true_obst,accs):

    # accs: 3x3 array of accuracy of each class. accs[0][2] is the probability of the preception returning car if there is nothing in the environment

    confs = [1/2,1/2]

    det_probs = accs[true_obst]
    labels = [0,1]
    det = random.choices(labels,weights = det_probs, k=1)[0]

    return det, confs



def stateUnsafe(obstClass,dist,vel):


    if obstClass == 0:
        if vel <= GOAL_CAR_SPEED-CAR_VEL_LOWER_BOUND_NOTHING or vel>=GOAL_CAR_SPEED+CAR_VEL_UPPER_BOUND_NOTHING:
            return True
        else:
            return False
    else:
        if vel <= CAR_VEL_LOWER_BOUND_CAR or vel >= CAR_VEL_UPPER_BOUND_CAR:
            return True
        if dist <= CAR_POS_FOLLOWING_LOWER_BOUND or dist >= CAR_POS_FOLLOWING_UPPER_BOUND:
            return True
        return False
    


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

def main(initSeeds = None,initNumTrajectories=100000):

    random.seed(23624)
    if initSeeds is None:
        numTrajectories = initNumTrajectories
    else:
        numTrajectories = len(initSeeds)

    numTrajectories = 1000#250

    episode_length = 60
    episode_length_PRISM = 30
    time_step = 0.1

    init_car_dist_base = 20 #FIXME: was 25
    init_car_vel = 8
    init_dist_noise = 2
    init_vel_noise = 1
    
    num_unsafe = 0


    max_dist = 100
    max_vel = 30

    dist_disc = 0.5
    vel_disc = 0.4

    ERRONEOUS_OBJ_DET_DIST = 10

    OBJ_DIST_NOISE = 1 ## FIXME: was 1
    OBJ_VEL_NOISE = 1 ## FIXME: was 1
    CAR_VEL_EST_NOISE = 1 ## FIXME: was 1

    w = World(init_car_dist_base, init_car_vel, "nothing", 0, dist_disc, vel_disc, episode_length, time_step)

    obsts_str = ["nothing","car"]
    obsts_vel = [0,8]
    
    fixed_accs = [[0.95,0.05],[0.05,0.95]]


    safe_prob_base_dir = "../models/knownControl3/safetyProbs/"


    dataSaveDir = "../results/AEBS_sim/run1/"
    plotSaveDir = dataSaveDir + "plots/"
    os.makedirs(plotSaveDir,exist_ok=True)

    allGTDists = []
    allGTVels = []
    allGTObsts = []

    allPFDists = []
    allPFVels = []
    allPFObsts = []

    max_dist = 50
    max_vel = 50

    allSafeProbs = []
    allSafeProbsStateDist = []
    allTrueSafeProbs = []
    allSafeUnsafe = []


    allTrialLengths = []

    plotMonitorCalibration = True
    plotEachTrial = False
    plotAUCCurve = True
    plotTrials = False

    for step in range(numTrajectories):


        print("Trial " + str(step),flush=True)

        true_obst = random.choice([0,1])
        # true_obst = 0

        # print("True obst: " + str(obsts_str[true_obst]))

        true_obst_str = obsts_str[true_obst]
        true_obst_vel = obsts_vel[true_obst]

        init_car_vel = (10 if true_obst == 0 else 8) + random.uniform(-init_vel_noise,init_vel_noise)
        init_car_dist = init_car_dist_base + random.uniform(-init_dist_noise,init_dist_noise)

        estimated_safety_probs = []
        estimated_safety_probs_state_dist = []
        true_safety_probs = []
        safe_unsafe_over_time = []


        w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

        pf = particleFilter(init_car_vel,fixed_accs,time_step,dist_disc,vel_disc,max_dist=max_dist,max_vel=max_vel)
        
        pf_pred_dist,pf_pred_vel,pf_pred_obst = pf.get_filter_state()

        # print("Initial states")
        # print("GT state: " + str([w.car_dist,w.car_vel-true_obst_vel]))
        # print("PF state: " + str([pf_pred_dist,pf_pred_vel]))

        gtDists = []
        gtVels = []

        ## give PF a few steps to warmup
        PF_warmup_steps = 5
        for _ in range(PF_warmup_steps):
            pred_obj_ind, _ = classification_model_fixed_acc(true_obst,fixed_accs)
            pred_obj = obsts_str[pred_obj_ind]

            if pred_obj == "nothing":
                obj_dist = 0
                obj_dist_mon = 15
                obj_vel = 0
            elif pred_obj == "car":
                if true_obst_str == "nothing":
                    obj_dist = ERRONEOUS_OBJ_DET_DIST
                    obj_dist_mon = obj_dist
                else:
                    obj_dist = w.car_dist
                    obj_dist_mon = obj_dist
                obj_vel = obsts_vel[1]
            
            obj_vel_mon = obj_vel + np.random.normal(0,OBJ_VEL_NOISE,1)[0]
            obj_dist_mon += np.random.normal(0,OBJ_DIST_NOISE,1)[0]
            car_vel_est = w.car_vel + np.random.normal(0,CAR_VEL_EST_NOISE,1)[0]

            # print("Perception: " + str([pred_obj_ind,obj_dist_mon,obj_vel_mon,car_vel_est]))

            ## run perception through filter and resample particles here
            _ = pf.step_filter_perception(pred_obj_ind,obj_dist_mon,obj_vel_mon,car_vel_est)


        ## keep track of previous 
        for e in range(episode_length):

            gtDists.append(w.car_dist)
            gtVels.append(w.car_vel-true_obst_vel)

            pred_obj_ind, _ = classification_model_fixed_acc(true_obst,fixed_accs)
            pred_obj = obsts_str[pred_obj_ind]

            if pred_obj == "nothing":
                obj_dist = 0
                obj_dist_mon = 15
                obj_vel = 0
            elif pred_obj == "car":
                if true_obst_str == "nothing":
                    obj_dist = ERRONEOUS_OBJ_DET_DIST
                    obj_dist_mon = obj_dist
                else:
                    obj_dist = w.car_dist
                    obj_dist_mon = obj_dist
                obj_vel = obsts_vel[1]
            
            obj_vel_mon = obj_vel + np.random.normal(0,OBJ_VEL_NOISE,1)[0]
            obj_dist_mon += np.random.normal(0,OBJ_DIST_NOISE,1)[0]
            car_vel_est = w.car_vel + np.random.normal(0,CAR_VEL_EST_NOISE,1)[0]

            # print("Perception: " + str([pred_obj_ind,obj_dist_mon,obj_vel_mon,car_vel_est]))

            ## run perception through filter and resample particles here
            particles = pf.step_filter_perception(pred_obj_ind,obj_dist_mon,obj_vel_mon,car_vel_est)

            pf_pred_dist,pf_pred_vel,pf_pred_obst = pf.get_filter_state()

            # print("GT state: " + str([w.car_dist,w.car_vel-true_obst_vel,true_obst]))
            # print("PF state: " + str([pf_pred_dist,pf_pred_vel,pf_pred_obst]))

            

            allGTDists.append(w.car_dist)
            allGTVels.append(w.car_vel-true_obst_vel)
            allGTObsts.append(true_obst)

            allPFDists.append(pf_pred_dist)
            allPFVels.append(pf_pred_vel)
            allPFObsts.append(pf_pred_obst)

            # car_dist, car_vel, crash, done, control_command = w.step(pred_obj, obj_dist, obj_vel, return_command=True)

            prev_dist = w.car_dist
            prev_vel = w.car_vel

            ## FIXME: use PF prediction
            car_dist, car_vel, crash, done, control_command = w.step_predVel(obsts_str[pf_pred_obst], pf_pred_dist, pf_pred_vel, return_command=True)

            ## FIXME: use actual state
            # car_dist, car_vel, crash, done, control_command = w.step_predVel(obsts_str[true_obst], w.car_dist, w.car_vel-true_obst_vel, return_command=True)


            #### compute safety probs
            ## ADDME: AEBS safety using state estimate and known control
            if stateUnsafe(pf_pred_obst,pf_pred_dist,pf_pred_vel+true_obst_vel):
                AEBSSafeProbStateEstThisTime = 0
            else:
                distInd = int(round(pf_pred_dist/dist_disc)) ## FIXME: should this be ceil or round????
                velInd = int(round((pf_pred_vel+true_obst_vel)/vel_disc))
                AEBSSafeProbStateEstThisTime = 1-computeViolationAEBSKnownControl(distInd,velInd,pf_pred_obst,int(control_command/vel_disc),safe_prob_base_dir)

            #### compute safety probs
            # print("Actual crash: " + str(crash))
            ## ADDME: AEBS safety using true state and known control
            # print("State: " + str([prev_dist,prev_vel,true_obst]))
            if stateUnsafe(true_obst,prev_dist,prev_vel):
                # print("Unsafe")
                AEBSSafeProbTrueStateThisTime = 0
            else:
                # print("Safe")
                distInd = round(prev_dist/dist_disc) ## FIXME: should this be ceil or round????
                velInd = round((prev_vel)/vel_disc)
                AEBSSafeProbTrueStateThisTime = 1-computeViolationAEBSKnownControl(distInd,velInd,true_obst,int(control_command/vel_disc),safe_prob_base_dir)
            # input("Wait")
            particles # pf particles this time
            AEBSSafeProbStateEstDistThisTime = 0
            numParticlesForEst = 0

            for particle in particles:
                particle_dist = particle[0]
                particle_vel = particle[1]+true_obst_vel
                particle_class = particle[2]

                if stateUnsafe(particle_class,particle_dist,particle_vel):
                    continue
                elif particle_class != pf_pred_obst:
                    continue
                else:
                    distInd = math.ceil(particle_dist/dist_disc) ## FIXME: should this be ceil or round????
                    velInd = math.ceil((particle_vel)/vel_disc)
                    AEBSSafeProbStateEstDistThisTime += 1-computeViolationAEBSKnownControl(distInd,velInd,particle_class,int(control_command/vel_disc),safe_prob_base_dir)
                    numParticlesForEst += 1
            
            if numParticlesForEst != 0:
                AEBSSafeProbStateEstDistThisTime = AEBSSafeProbStateEstDistThisTime/numParticlesForEst
            assert AEBSSafeProbStateEstDistThisTime >= 0 and AEBSSafeProbStateEstDistThisTime < 1.000001

            estimated_safety_probs.append(AEBSSafeProbStateEstThisTime)
            estimated_safety_probs_state_dist.append(AEBSSafeProbStateEstDistThisTime)
            true_safety_probs.append(AEBSSafeProbTrueStateThisTime)
            safe_unsafe_over_time.append(crash)





            _ = pf.step_filter_dynamics(control_command)




            if done:
                if crash:
                    num_unsafe += 1
                    print("Unsafe")
                # input("wait")
                break
        
        gtDists.append(w.car_dist)
        gtVels.append(w.car_vel-true_obst_vel)
        
        allTrialLengths.append(len(estimated_safety_probs))

        for time,safeProb in enumerate(estimated_safety_probs[0:episode_length]):
            allSafeProbs.append(safeProb)
            allSafeProbsStateDist.append(estimated_safety_probs_state_dist[time])
            allTrueSafeProbs.append(true_safety_probs[time])

            if sum(safe_unsafe_over_time[time:time+episode_length_PRISM]) >= 1:
                crash_this_time = 1
            else:
                crash_this_time = 0
            allSafeUnsafe.append(crash_this_time)



    print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))
    print('average length of scenario: ' + str(np.mean(allTrialLengths)))

    ## check for calibration
    # bin at level of 0.1
    num_bins = 10
    binned_counts = {}
    binned_counts_state_dist = {}
    binned_counts_true = {}
    for bin in range(num_bins):
        binned_counts[bin] = [0,0]
        binned_counts_state_dist[bin] = [0,0]
        binned_counts_true[bin] = [0,0]

    for i in range(len(allSafeProbs)):
        conf = allSafeProbs[i]
        confStateDist = allSafeProbsStateDist[i]
        confTrue = allTrueSafeProbs[i]
        safeUnsafe = allSafeUnsafe[i] # 0 if safe, 1 if unsafe

        bin = get_bin(conf,num_bins)
        # print("conf: " + str(confs[j]) + ", bin: " + str(bin))
        binned_counts[bin][0]+=safeUnsafe
        binned_counts[bin][1]+=1

        bin = get_bin(confStateDist,num_bins)
        binned_counts_state_dist[bin][0]+=safeUnsafe
        binned_counts_state_dist[bin][1]+=1

        bin = get_bin(confTrue,num_bins)
        binned_counts_true[bin][0]+=safeUnsafe
        binned_counts_true[bin][1]+=1


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


        # Plot ROC curve
        plt.clf()
        plt.plot(fpr, tpr, 'b-', label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot(fpr_state_dist, tpr_state_dist, 'g-', label='ROC curve (area = %0.3f)' % roc_auc_state_dist)
        plt.plot(fpr_GT_state, tpr_GT_state, 'r-', label='ROC curve (area = %0.3f)' % roc_auc_GT_state)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(plotSaveDir + "/safetyMonitorAUCCurve.png")










if __name__ == '__main__':
    main()
