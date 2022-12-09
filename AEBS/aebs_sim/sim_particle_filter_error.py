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




def computeViolationFromDistanceSpeedObst(d,v,obst_type_str,obst_type_int,dist_disc,vel_disc,accs,toPrint=True):

    if v == 12 and obst_type_int == 2:
        v=11.6
        
    if int(v/vel_disc) <= 0:
        if toPrint:
            print("Car stopped")
        if obst_type_int == 1:
            if toPrint:
                print("Violation prob: " + str(0))
            return 0
        else:
            if toPrint:
                print("Violation prob: " + str(1))
            return 1
    if int(d/dist_disc) <= 0:
        if obst_type_int == 1 or obst_type_int == 2:
            if toPrint:
                print("Violation prob: " + str(1))
            return 1
    if obst_type_int == 0:
        folder_name = "../../../results/AEBS/violationPolys/vel_" + str(int(v/vel_disc)) + "/"
    else:
        folder_name = "../../../results/AEBS/violationPolys/dist_" + str(int(d/dist_disc)) + "_vel_" + str(int(v/vel_disc)) + "/"
    file_name  = folder_name + "violationPoly" + str(obst_type_str.capitalize()) + ".txt"

    if toPrint:
        print(file_name)
    # print(obst_type_int)
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
                # print("Replacing car")
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
                if toPrint:
                    print("Violation prob: " + str(violation_prob))
                # print(violation_prob)

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




def main(initSeeds = None,initNumTrajectories=100000):

    random.seed(23624)
    if initSeeds is None:
        numTrajectories = initNumTrajectories
    else:
        numTrajectories = len(initSeeds)

    numTrajectories = 1000#250

    episode_length = 60
    time_step = 0.1

    init_car_dist = 25
    init_car_vel = 8
    num_unsafe = 0

    max_dist = 65
    max_vel = 12

    dist_disc = 0.5
    vel_disc = 0.4

    ERRONEOUS_OBJ_DET_DIST = 10

    OBJ_DIST_NOISE = 1
    OBJ_VEL_NOISE = 1

    w = World(init_car_dist, init_car_vel, "nothing", 0, dist_disc, vel_disc, episode_length, time_step)

    obsts_str = ["nothing","rock","car"]
    obsts_vel = [0,0,8]
    
    fixed_accs = [[0.5,0.25,0.25],[0.4,0.2,0.4],[0.2,0.5,0.3]]
    
    allObsts = []


    plotSaveDir = "../results/pfDataForModeling/"
    os.makedirs(plotSaveDir,exist_ok=True)

    allGTDists = []
    allGTVels = []
    allGTObsts = []

    allPFDists = []
    allPFVels = []
    allPFObsts = []


    for step in range(numTrajectories):

        print("Trial " + str(step),flush=True)

        true_obst = random.choice([0,1,2])
        # true_obst = 1

        print("True obst: " + str(obsts_str[true_obst]))

        true_obst_str = obsts_str[true_obst]
        true_obst_vel = obsts_vel[true_obst]

        w.reset(init_car_dist, init_car_vel, true_obst_str, true_obst_vel)

        all_preds = []
        pred_counts = [0,0,0]

        all_confs = []
        dist_speed_safety_probs = []
        true_safety_probs = []

        car_dist = init_car_dist
        car_vel = init_car_vel

        pf = particleFilter(init_car_vel,fixed_accs,time_step,dist_disc,vel_disc,max_dist=max_dist,max_vel=max_vel)
        
        ## keep track of previous 
        for e in range(episode_length):


            pred_obj_ind, confs = classification_model_fixed_acc(true_obst,fixed_accs)
            pred_counts[pred_obj_ind]+=1

            pred_obj = obsts_str[pred_obj_ind]

            all_preds.append(pred_obj)
            all_confs.append(confs)

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
            
            obj_vel_mon = obj_vel + np.random.normal(0,OBJ_VEL_NOISE,1)[0]
            obj_dist_mon += np.random.normal(0,OBJ_DIST_NOISE,1)[0]

            pf_pred_dist,pf_pred_vel,pf_pred_obst = pf.get_filter_state()

            # print("True state: " + str([w.car_dist,w.car_vel-true_obst_vel,true_obst]))
            # print("PF state: " + str([pf_pred_dist,pf_pred_vel,pf_pred_obst]))
            # car_dist, car_vel, crash, done, control_command = w.step(pred_obj, obj_dist, obj_vel, return_command=True)
            car_dist, car_vel, crash, done, control_command = w.step_predVel(obsts_str[pf_pred_obst], pf_pred_dist, pf_pred_vel, return_command=True)

            # print("Control Command: " + str(control_command))

            particles = pf.step_filter(pred_obj_ind,obj_dist_mon,obj_vel_mon,car_vel,control_command)
            pf_pred_dist,pf_pred_vel,pf_pred_obst = pf.get_filter_state()

            allGTDists.append(car_dist)
            allGTVels.append(car_vel)
            allGTObsts.append(true_obst)

            allPFDists.append(pf_pred_dist)
            allPFVels.append(pf_pred_vel)
            allPFObsts.append(pf_pred_obst)

            ## violation/safety prob from pf states
            violation_prob = 0
            for particle in particles:
                dist_speed_violation_prob = computeViolationFromDistanceSpeedObst(particle[0],particle[1]-obj_vel,obsts_str[particle[2]],particle[2],dist_disc,vel_disc,fixed_accs,toPrint=False)
                violation_prob += dist_speed_violation_prob/len(particles)
            
            if violation_prob<0 and violation_prob>-0.0001:
                violation_prob = 0
            if violation_prob>1 and violation_prob<1.0001:
                violation_prob = 1
            assert violation_prob>=0 and violation_prob<=1
            safe_prob = 1-violation_prob

            assert safe_prob >=0 and safe_prob <= 1
            dist_speed_safety_probs.append(safe_prob)

            ## violation/safety prob from true state
            true_violation_prob = computeViolationFromDistanceSpeedObst(w.car_dist,car_vel,true_obst_str,true_obst,dist_disc,vel_disc,fixed_accs,toPrint=False)
            if violation_prob<0 and violation_prob>-0.0001:
                violation_prob = 0
            if violation_prob>1 and violation_prob<1.0001:
                violation_prob = 1
            assert violation_prob>=0 and violation_prob<=1
            true_safe_prob = 1-true_violation_prob
            true_safety_probs.append(true_safe_prob)


            if done:
                if crash:
                    num_unsafe += 1
                break
                    
        # print("Safety probs")
        # print(dist_speed_safety_probs)

        # print("True safety probs")
        # print(true_safety_probs)
        
    print('number of crashes: ' + str(num_unsafe) + ', ' + str(num_unsafe/numTrajectories))


    ## save pf data for processing
    with open(plotSaveDir + "trueDists.pkl",'wb') as f:
        pickle.dump(allGTDists,f)
    with open(plotSaveDir + "trueVels.pkl",'wb') as f:
        pickle.dump(allGTVels,f)
    with open(plotSaveDir + "trueObsts.pkl",'wb') as f:
        pickle.dump(allGTObsts,f)
    with open(plotSaveDir + "pfDists.pkl",'wb') as f:
        pickle.dump(allPFDists,f)
    with open(plotSaveDir + "pfVels.pkl",'wb') as f:
        pickle.dump(allPFVels,f)
    with open(plotSaveDir + "pfObsts.pkl",'wb') as f:
        pickle.dump(allPFObsts,f)









if __name__ == '__main__':
    main()
