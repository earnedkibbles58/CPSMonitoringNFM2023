import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import math
import pickle



PLOTERRORHISTS = True

def customRound(val,discretization):

    return math.ceil(val/discretization)

def main():

    dist_disc = 0.25 ## FIXME: had been 0.5
    vel_disc = 0.4

    # dataSaveDir = "../results/pfDataForModeling/noObst_28/"
    dataSaveDir = "../results/pfDataForModeling/carObst_28/"


    ## save pf data for processing
    with open(dataSaveDir + "trueDists.pkl",'rb') as f:
        allGTDists = pickle.load(f)
    with open(dataSaveDir + "trueVels.pkl",'rb') as f:
        allGTVels = pickle.load(f)
    with open(dataSaveDir + "trueObsts.pkl",'rb') as f:
        allGTObsts = pickle.load(f)
    with open(dataSaveDir + "pfDists.pkl",'rb') as f:
        allPFDists = pickle.load(f)
    with open(dataSaveDir + "pfVels.pkl",'rb') as f:
        allPFVels = pickle.load(f)
    with open(dataSaveDir + "pfObsts.pkl",'rb') as f:
        allPFObsts = pickle.load(f)
    with open(dataSaveDir + "percentCrash.pkl",'rb') as f:
        numCrashAndTrials = pickle.load(f)
    numCrash = numCrashAndTrials[0]
    numTrials = numCrashAndTrials[1]

    print("percent crash: " + str(float(numCrash/numTrials)) + ", " + str(numCrash) + "/" + str(numTrials))


    numClasses = 2

    assert len(allGTDists) == len(allPFDists)

    for c in range(numClasses):
        distErrs = []
        velErrs = []

        for i in range(len(allGTDists)):
            if allGTObsts[i] != c:
                continue
            if allGTObsts[i]==1 and allPFObsts[i]==0:
                continue
            distErrs.append(allPFDists[i]-allGTDists[i])
            velErrs.append(allPFVels[i]-allGTVels[i])

        distErrDict = dict()
        velErrDict = dict()

        for i in range(len(distErrs)):
            # distErr = round(distErrs[i])
            distErr = customRound(distErrs[i],dist_disc)
            if distErr in distErrDict:
                distErrDict[distErr] += 1/len(distErrs)
            else:
                distErrDict[distErr] = 1/len(distErrs)
            
            # velErr = round(velErrs[i])
            velErr = customRound(velErrs[i],vel_disc)
            if velErr in velErrDict:
                velErrDict[velErr] += 1/len(velErrs)
            else:
                velErrDict[velErr] = 1/len(velErrs)

        print("True class " + str(c) + " model")
        if c==1:
            print("Dist Errs: " + str(distErrDict))
        print("Vel Errs: " + str(velErrDict))


    confusionMatrix = np.zeros([numClasses,numClasses])
    for i in range(len(allGTDists)):
        confusionMatrix[allGTObsts[i]][allPFObsts[i]] += 1
    print("Confusion Matrix: " + str(confusionMatrix))




def main_diffDistVelErrsPerDet():
    dist_disc = 0.25 ## FIXME: had been 0.25
    vel_disc = 0.4

    otherCarSpeed = 8

    # dataSaveDir = "../results/pfDataForModeling/noObst_28/"
    # dataSaveDir = "../results/pfDataForModeling/carObst_28/"
    dataSaveDir = "../results/pfDataForModeling/noObst_29_distDisc0.25/"

    plotSaveDir = "../results/AEBS_sim/run1_bugFixCrashVar_carObst_noObstDet_distDisc0.25/PFErrPlots/"
    os.makedirs(plotSaveDir,exist_ok=True)


    ## save pf data for processing
    with open(dataSaveDir + "trueDists.pkl",'rb') as f:
        allGTDists = pickle.load(f)
    with open(dataSaveDir + "trueVels.pkl",'rb') as f:
        allGTVels = pickle.load(f)
    with open(dataSaveDir + "trueObsts.pkl",'rb') as f:
        allGTObsts = pickle.load(f)
    with open(dataSaveDir + "pfDists.pkl",'rb') as f:
        allPFDists = pickle.load(f)
    with open(dataSaveDir + "pfVels.pkl",'rb') as f:
        allPFVels = pickle.load(f)
    with open(dataSaveDir + "pfObsts.pkl",'rb') as f:
        allPFObsts = pickle.load(f)
    with open(dataSaveDir + "percentCrash.pkl",'rb') as f:
        numCrashAndTrials = pickle.load(f)
    numCrash = numCrashAndTrials[0]
    numTrials = numCrashAndTrials[1]

    print("percent crash: " + str(float(numCrash/numTrials)) + ", " + str(numCrash) + "/" + str(numTrials))


    numClasses = 2

    assert len(allGTDists) == len(allPFDists)

    for gtObst in range(numClasses):

        for predObst in range(numClasses):
            print("True class " + str(gtObst) + " and pred class " + str(predObst))

            distErrs = []
            velErrs = []

            for i in range(len(allGTDists)):
                if allGTObsts[i] != gtObst:
                    continue
                if allPFObsts[i] != predObst:
                    continue
                # if allGTObsts[i]==1 and allPFObsts[i]==0:
                #     continue
                distErrs.append(allPFDists[i]-allGTDists[i])
                velErrs.append(allPFVels[i]-allGTVels[i])

            distErrDict = dict()
            velErrDict = dict()

            for i in range(len(distErrs)):
                # distErr = round(distErrs[i])
                distErr = customRound(distErrs[i],dist_disc)
                if distErr in distErrDict:
                    distErrDict[distErr] += 1/len(distErrs)
                else:
                    distErrDict[distErr] = 1/len(distErrs)
                
                # velErr = round(velErrs[i])
                velErr = customRound(velErrs[i],vel_disc)
                if gtObst == 1 and predObst == 0:
                    velErr = velErr - otherCarSpeed
                if velErr in velErrDict:
                    velErrDict[velErr] += 1/len(velErrs)
                else:
                    velErrDict[velErr] = 1/len(velErrs)

            print("True class " + str(gtObst) + " and pred class " + str(predObst) + " model")
            if predObst==1:
                print("Dist Errs: " + str(distErrDict))
            print("Vel Errs: " + str(velErrDict))


    confusionMatrix = np.zeros([numClasses,numClasses])
    for i in range(len(allGTDists)):
        confusionMatrix[allGTObsts[i]][allPFObsts[i]] += 1
    print("Confusion Matrix: " + str(confusionMatrix))

    if PLOTERRORHISTS:

        # dist_errs = [] #[allPFDists[i]-allGTDists[i] for i in range(len(allGTDists))]
        # vel_errs = [allPFVels[i]-allGTVels[i] for i in range(len(allGTVels))]

        # for i in range(len(allPFDists)):
        #     dist_err = allPFDists[i]-allGTDists[i]
        #     # if abs(dist_err) > 50:
        #     #     continue
        #     dist_errs.append(dist_err)

        pfDistErrs = [allPFDists[i]-allGTDists[i] for i in range(len(allPFDists))]
        pfVelErrs = [allPFVels[i]-allGTVels[i] for i in range(len(allPFVels))]

        min_dist_bin = math.floor(min(pfDistErrs)/dist_disc)
        max_dist_bin = math.ceil(max(pfDistErrs)/dist_disc)
        dist_bins = np.arange(min_dist_bin,max_dist_bin,dist_disc) #[i for i in range(min_dist_bin,max_dist_bin,dist_disc)]

        plt.clf()
        plt.hist(pfDistErrs,bins=dist_bins,edgecolor = "black")
        plt.xlabel("State Estimator Distance Error")
        plt.ylabel("Bin Counts")
        # plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/pfDistErrorModel.png")


        min_vel_bin = math.floor(min(pfVelErrs)/vel_disc)
        max_vel_bin = math.ceil(max(pfVelErrs)/vel_disc)
        vel_bins = np.arange(min_vel_bin,max_vel_bin,vel_disc) #[i for i in range(min_vel_bin,max_vel_bin,vel_disc)]
        plt.clf()
        plt.hist(pfVelErrs,bins=vel_bins,edgecolor = "black")
        plt.xlabel("State Estimator Velocity Error")
        plt.ylabel("Bin Counts")
        # plt.xlim([0, 1])
        # plt.ylim([-0.1, 1.1])
        plt.savefig(plotSaveDir + "/pfVelErrorModel.png")


if __name__ == '__main__':
    # main()
    main_diffDistVelErrsPerDet()
