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
import pickle

from testCalibratedModel import calibrated_perception
from runtime_monitor import prismViolationProbNothing, prismViolationProbRock, prismViolationProbCar
from car_monitor_particle_filter import particleFilter

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import brier_score_loss






def classification_model_fixed_acc(true_obst,accs):

    # accs: 3x3 array of accuracy of each class. accs[0][2] is the probability of the preception returning car if there is nothing in the environment

    confs = [1/2,1/2]

    det_probs = accs[true_obst]
    labels = [0,1]
    det = random.choices(labels,weights = det_probs, k=1)[0]

    return det, confs







def main(initSeeds = None,initNumTrajectories=100000):

    random.seed(23624)
    if initSeeds is None:
        numTrajectories = initNumTrajectories
    else:
        numTrajectories = len(initSeeds)

    trialNumLow = 0
    numTrajectories = 1000#250

    episode_length = 60
    time_step = 0.1

    init_car_dist = 20 #FIXME: was 25
    init_car_vel = 8
    num_unsafe = 0

    max_dist = 100
    max_vel = 30

    dist_disc = 0.5
    vel_disc = 0.4

    ERRONEOUS_OBJ_DET_DIST = 10

    OBJ_DIST_NOISE = 1 ## FIXME: was 1
    OBJ_VEL_NOISE = 1 ## FIXME: was 1
    CAR_VEL_EST_NOISE = 1 ## FIXME: was 1

    w = World(init_car_dist, init_car_vel, "nothing", 0, dist_disc, vel_disc, episode_length, time_step)

    obsts_str = ["nothing","car"]
    obsts_vel = [0,8]
    
    fixed_accs = [[0.95,0.05],[0.05,0.95]]
    

    dataSaveDir = "../results/pfDataForModeling/noObst_28/"
    # dataSaveDir = "../results/pfDataForModeling/carObst_28/"
    os.makedirs(dataSaveDir,exist_ok=True)

    # plotSaveDir = "../results/pfDebugging/noObst/"
    # trialSaveDir = plotSaveDir + "trials/"
    # pfSaveDirBase = plotSaveDir + "pfPlots/"
    # os.makedirs(trialSaveDir,exist_ok=True)
    # os.makedirs(trialSaveDir,exist_ok=True)

    allGTDists = []
    allGTVels = []
    allGTObsts = []

    allPFDists = []
    allPFVels = []
    allPFObsts = []

    max_dist = 50
    max_vel = 50


    for step in range(trialNumLow, trialNumLow+numTrajectories):

        # print("Trial " + str(step),flush=True)

        # true_obst = random.choice([0,1])
        true_obst = 0

        # print("True obst: " + str(obsts_str[true_obst]))

        true_obst_str = obsts_str[true_obst]
        true_obst_vel = obsts_vel[true_obst]

        init_car_vel = 10 if true_obst == 0 else 8

        w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

        car_dist = init_car_dist
        car_vel = init_car_vel

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

            # print("GT state: " + str([w.car_dist,w.car_vel-true_obst_vel]))
            # print("PF state: " + str([pf_pred_dist,pf_pred_vel]))

            

            allGTDists.append(w.car_dist)
            allGTVels.append(w.car_vel-true_obst_vel)
            allGTObsts.append(true_obst)

            allPFDists.append(pf_pred_dist)
            allPFVels.append(pf_pred_vel)
            allPFObsts.append(pf_pred_obst)

            # car_dist, car_vel, crash, done, control_command = w.step(pred_obj, obj_dist, obj_vel, return_command=True)

            ## FIXME: use PF prediction
            car_dist, car_vel, crash, done, control_command = w.step_predVel(obsts_str[pf_pred_obst], pf_pred_dist, pf_pred_vel, return_command=True)

            ## FIXME: use actual state
            # car_dist, car_vel, crash, done, control_command = w.step_predVel(obsts_str[true_obst], w.car_dist, w.car_vel-true_obst_vel, return_command=True)


            # print("GT state: " + str([w.car_dist,w.car_vel-true_obst_vel]))



            _ = pf.step_filter_dynamics(control_command)

            if done:
                if crash:
                    num_unsafe += 1
                break
        
        gtDists.append(w.car_dist)
        gtVels.append(w.car_vel-true_obst_vel)
                    
        ## plot pf states over time
        # pfSaveDir = pfSaveDirBase + "/trial" + str(step)
        # os.makedirs(pfSaveDir,exist_ok=True)
        # os.makedirs(pfSaveDir+"/carStates",exist_ok=True)
        # os.makedirs(pfSaveDir+"/obstVels",exist_ok=True)
        # os.makedirs(pfSaveDir+"/obstClasses",exist_ok=True)
        # pf.plot_filter_states(pfSaveDir,carDists=gtDists,carVels=gtVels)

    print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))


    ## save pf data for processing
    with open(dataSaveDir + "trueDists.pkl",'wb') as f:
        pickle.dump(allGTDists,f)
    with open(dataSaveDir + "trueVels.pkl",'wb') as f:
        pickle.dump(allGTVels,f)
    with open(dataSaveDir + "trueObsts.pkl",'wb') as f:
        pickle.dump(allGTObsts,f)
    with open(dataSaveDir + "pfDists.pkl",'wb') as f:
        pickle.dump(allPFDists,f)
    with open(dataSaveDir + "pfVels.pkl",'wb') as f:
        pickle.dump(allPFVels,f)
    with open(dataSaveDir + "pfObsts.pkl",'wb') as f:
        pickle.dump(allPFObsts,f)
    with open(dataSaveDir + "percentCrash.pkl",'wb') as f:
        pickle.dump([num_unsafe,numTrajectories],f)


    # print("Maximum distance: " + str(max_dist))
    # print("Maximum velocity: " + str(max_vel))






if __name__ == '__main__':
    main()
