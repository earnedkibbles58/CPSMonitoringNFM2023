import os
import subprocess
import time

wlMax = 100
deltawl = 2
wlidMax = int(wlMax/deltawl)+1
trimming = True

counter = 0
numProc = 50

for contAction1 in range(0,2):
    for contAction2 in range(0,2):
        for wl1 in range(1,wlidMax+1):
            for wl2 in range(1,wl1+1):

                if contAction1 == 0 and contAction2 == 0:
                    continue

                print("Running " + str([wl1,wl2,contAction1,contAction2]) + " out of " + str([wlidMax]), flush=True)

                if trimming:
                    dir_to_save = "../../models/safetyProbs/withTrimming/if13.5_of4.3_deltawl" + str(deltawl) + "/contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "/wl1_" + str(wl1) + "_wl2_" + str(wl2) + "/"
                    prism_model = "../../models/withTrimming/if13.5_of4.3_deltawl" + str(deltawl) + "/waterTankBaseline.prism"
                else:
                    dir_to_save = "../../models/safetyProbs/noTrimming/if13.5_of4.3_deltawl" + str(deltawl) + "/contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "/wl1_" + str(wl1) + "_wl2_" + str(wl2) + "/"
                    prism_model = "../../models/noTrimming/if13.5_of4.3_deltawl" + str(deltawl) + "/waterTankBaseline.prism"
                os.makedirs(dir_to_save,exist_ok=True)
                file_to_save = dir_to_save + "/violationProb.txt"

                props_file = "../../models/tankPropsSink.props"


                with open(file_to_save,'w') as f:
                    process = subprocess.Popen(["nice", "-n 10", "/data2/mcleav/safetyContracts/ICCPS_2022/prism/prism-4.5/prism/bin/prism", prism_model, props_file, "-const", "wlidInit1=" + str(wl1) + ",wlidInit2=" + str(wl2) + ",initContAction1=" + str(contAction1) + ",initContAction2=" + str(contAction2), "-javamaxmem", "16g"],stdout=f)

                # out, err = process.communicate()
                # print(out)

                # with open(file_to_save,'w') as file:
                #     file.write(out.decode("utf-8"))

                counter +=1

                if counter % numProc == numProc-1:
                    print("Sleeping for 10 seconds")
                    time.sleep(10)
                    print("Done sleeping")

