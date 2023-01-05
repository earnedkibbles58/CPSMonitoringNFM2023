import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import math
import pickle




def customRound(val,discretization):

    return math.ceil(val/discretization)

def main():

    dist_disc = 0.5
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
    dist_disc = 0.5
    vel_disc = 0.4

    otherCarSpeed = 8

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

if __name__ == '__main__':
    # main()
    main_diffDistVelErrsPerDet()
