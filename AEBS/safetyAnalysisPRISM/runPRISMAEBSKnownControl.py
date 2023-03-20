import os
import subprocess
import time
import math

distDisc = 0.25 ## FIXME: had been 0.5
velDisc = 0.4

controlActions = [0,10,-10]

fakeInitPosNoObst = 40

noObstSpeedLower = 8.5
noObstSpeedUpper = 11.5

velIndLower = math.floor(noObstSpeedLower/velDisc)
velIndUpper = math.ceil(noObstSpeedUpper/velDisc)

counter = 0
numProc = 20
initSeqFlag = 2
sleepTime = 60

verifNoObst = False
verifCarObst = True

if verifNoObst:
    # no obst
    for controlAction in controlActions:
        for velInd in range(velIndLower,velIndUpper+1):

            print("No obst, running " + str([velInd,controlAction]),flush=True)

            dir_to_save = "../models/knownControl3_noMisDets/safetyProbs/noObst/contAction_" + str(controlAction) + "/initVelInd_" + str(velInd) + "/"
            prism_model = "../models/knownControl3_noMisDets/carModel_DistDisc" + str(distDisc) + "_nothing.prism"
            os.makedirs(dir_to_save,exist_ok=True)
            file_to_save = dir_to_save + "/violationProb.txt"

            props_file = "../models/knownControl3/props_nothing.props"

            with open(file_to_save,'w') as f:
                process = subprocess.Popen(["nice", "-n 10", "/data2/mcleav/safetyContracts/ParamPerContracts/caseStudy/prism/prism-4.5/prism/bin/prism", prism_model, props_file, "-const", "initPos=" + str(fakeInitPosNoObst) + ",initSpeed=" + str(velInd) + ",initCont=" + str(controlAction) + ",initState=" + str(initSeqFlag), "-javamaxmem", "16g"],stdout=f)


            counter += 1
            if counter % numProc == numProc-1:
                counter = 0
                print("Sleeping for " + str(sleepTime) + " seconds",flush=True)
                time.sleep(sleepTime)
                print("Done sleeping")




if verifCarObst:
    # car obst
    carObstSpeedLower = 6
    carObstSpeedUpper = 12

    velIndLower = math.floor(carObstSpeedLower/velDisc)
    velIndUpper = math.ceil(carObstSpeedUpper/velDisc)

    carDistLower = 15
    carDistUpper = 25

    distIndLower = math.floor(carDistLower/distDisc)
    distIndUpper = math.floor(carDistUpper/distDisc)

    # car obst
    for controlAction in controlActions:
        for velInd in range(velIndLower,velIndUpper+1):
            for distInd in range(distIndLower,distIndUpper+1):

                print("Car obst, running " + str([distInd,velInd,controlAction]),flush=True)

                dir_to_save = "../models/knownControl3_noMisDets/safetyProbs_DistDisc" + str(distDisc) + "/carObst/contAction_" + str(controlAction) + "/initVelInd_" + str(velInd) + "_initDistInd_" + str(distInd) + "/"
                prism_model = "../models/knownControl3_noMisDets/carModel_DistDisc" + str(distDisc) + "_car.prism"
                os.makedirs(dir_to_save,exist_ok=True)
                file_to_save = dir_to_save + "/violationProb.txt"

                props_file = "../models/knownControl3/props_car_distDisc" + str(distDisc) + ".props"

                with open(file_to_save,'w') as f:
                    process = subprocess.Popen(["nice", "-n 10", "/data2/mcleav/safetyContracts/ParamPerContracts/caseStudy/prism/prism-4.5/prism/bin/prism", prism_model, props_file, "-const", "initPos=" + str(distInd) + ",initSpeed=" + str(velInd) + ",initCont=" + str(controlAction) + ",initState=" + str(initSeqFlag), "-javamaxmem", "16g"],stdout=f)


                counter += 1
                if counter % numProc == numProc-1:
                    counter = 0
                    print("Sleeping for " + str(sleepTime) + " seconds",flush=True)
                    time.sleep(sleepTime)
                    print("Done sleeping")

print("DONE!")