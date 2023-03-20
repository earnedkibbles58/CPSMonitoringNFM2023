import os
import subprocess
import time

wlMax = 100
deltawl = 1
wlidMax = int(wlMax/deltawl)+1
trimming = True

counter = 0
numProc = 20
initCurrN = 2

for contAction1 in range(0,2):
    for contAction2 in range(0,2):
        for globalAction1 in range(0,2):
            for globalAction2 in range(0,2):
                for wl1 in range(1,wlidMax+1):
                    for wl2 in range(1,wl1+1):

                        # ignore infeasible configurations
                        if contAction1 == 0 and globalAction1 == 1:
                            continue
                        if contAction2 == 0 and globalAction2 == 1:
                            continue
                        if globalAction1 == 1 and globalAction2 == 1:
                            continue
                        if contAction1 == 1 and contAction2 == 0 and globalAction1 == 0:
                            continue
                        if contAction1 == 0 and contAction2 == 1 and globalAction2 == 0:
                            continue
                        if contAction1 == 1 and globalAction1 == 0 and globalAction2 == 0:
                            continue
                        if contAction2 == 1 and globalAction1 == 0 and globalAction2 == 0:
                            continue

                        # this case is running in another screen
                        if not (contAction1 == 1 and contAction2 == 1 and globalAction1 == 1 and globalAction2 == 0):
                            continue

                        if not wl1 == 63:
                            continue
                        if wl2 < 41:
                            continue
                        if wl2 > 59:
                            continue

                        print("Running " + str([wl1,wl2,contAction1,contAction2,globalAction1,globalAction2]) + " out of " + str([wlidMax]), flush=True)

                        if trimming:
                            dir_to_save = "../../models/safetyProbs/knownControl/withTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "_globalAction1_" + str(globalAction1) + "_globalAction2_" + str(globalAction2) + "/wl1_" + str(wl1) + "_wl2_" + str(wl2) + "/"
                            prism_model = "../../models/knownControl/withTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/waterTankBaseline.prism"
                        else:
                            dir_to_save = "../../models/safetyProbs/knownControl/noTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "_globalAction1_" + str(globalAction1) + "_globalAction2_" + str(globalAction2) + "/wl1_" + str(wl1) + "_wl2_" + str(wl2) + "/"
                            prism_model = "../../models/knownControl/noTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/waterTankBaseline.prism"
                        os.makedirs(dir_to_save,exist_ok=True)
                        file_to_save = dir_to_save + "/violationProb.txt"

                        props_file = "../../models/tankPropsSink.props"


                        with open(file_to_save,'w') as f:
                            process = subprocess.Popen(["nice", "-n 10", "/data2/mcleav/safetyContracts/ICCPS_2022/prism/prism-4.5/prism/bin/prism", prism_model, props_file, "-const", "wlidInit1=" + str(wl1) + ",wlidInit2=" + str(wl2) + ",initContAction1=" + str(contAction1) + ",initContAction2=" + str(contAction2) + ",initContActionG1=" + str(globalAction1) + ",initContActionG2=" + str(globalAction2) + ",initCurrN=" + str(initCurrN), "-javamaxmem", "16g"],stdout=f)

                        # out, err = process.communicate()
                        # print(out)

                        # with open(file_to_save,'w') as file:
                        #     file.write(out.decode("utf-8"))


                        # if wl1 >= 40 and wl1 <= 70 and wl2>=40 and contAction1==1 and contAction2==1:
                        #     counter = min(counter+5,numProc-1)
                        # else:
                            # counter +=1

                        counter += 1
                        sleepTime = 150
                        if counter % numProc == numProc-1:
                            counter = 0
                            print("Sleeping for " + str(sleepTime) + " seconds")
                            time.sleep(sleepTime)
                            print("Done sleeping")

