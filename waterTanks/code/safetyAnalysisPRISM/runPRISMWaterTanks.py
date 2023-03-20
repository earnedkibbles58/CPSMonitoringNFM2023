import os
import subprocess
import time

wlMax = 100
deltawl = 1
wlidMax = int(wlMax/deltawl)+1
trimming = True

counter = 0
numProc = 30

for contAction1 in range(0,2):
    for contAction2 in range(0,2):
        for wl1 in range(1,wlidMax+1):
            for wl2 in range(1,wl1+1):

                if contAction1 == 0 or contAction2 == 0:
                    continue

                # if contAction1 == 1 and contAction2 == 1:
                #     continue

                if wl1<=69:
                    continue
                if wl1 >= 40 and wl1 <= 60 and wl2>=40 and contAction1==1 and contAction2==1:
                    continue # only do the easy ones for this one
                if wl1 == 70 and (wl2<46 or wl2>50):
                    continue
                if wl1 == 71 and (wl2<36 or wl2>55):
                    continue
                if wl1 == 72 and (wl2<36 or wl2>57):
                    continue
                if wl1 == 73 and (wl2<37 or wl2>58):
                    continue
                if wl1 == 74 and (wl2<37 or wl2>64):
                    continue
                if wl1 == 75 and (wl2<32 or wl2>67):
                    continue
                if wl1 == 76 and (wl2<31 or wl2>69):
                    continue
                if wl1 == 77 and (wl2<31 or wl2>70):
                    continue
                if wl1 == 78 and (wl2<31 or wl2>77):
                    continue
                if wl1 == 79 and (wl2<30 or wl2>78):
                    continue
                if wl1 == 80 and wl2<23:
                    continue
                
                if wl1<88:
                    continue
                if wl1==88 and wl2<39:
                    continue

                ## 60-69 running on runPRISMWaterTankWL1
                # if wl1 < 60 or wl1 >= 70:
                #     continue
                # if wl1 == 60 and wl2<43:
                #     continue
                # if wl1==61 and wl2<39:
                #     continue
                # if wl1==62 and wl2<37:
                #     continue

                ## 55-59 running on runPRISMWaterTankWL1_2
                # if wl1 < 55 or wl1 >= 60:
                #     continue
                # if wl1 == 55 and wl2<45 and wl2 >29:
                #     continue
                # if wl1 == 56 and wl2<45:
                #     continue
                # if wl1 == 57 and wl2<43:
                #     continue
                # if wl1 == 58 and wl2<41:
                #     continue
                # if wl1 == 59 and wl2<38:
                #     continue


                print("Running " + str([wl1,wl2,contAction1,contAction2]) + " out of " + str([wlidMax]), flush=True)

                if trimming:
                    dir_to_save = "../../models/safetyProbs/withTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "/wl1_" + str(wl1) + "_wl2_" + str(wl2) + "/"
                    prism_model = "../../models/withTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/waterTankBaseline.prism"
                else:
                    dir_to_save = "../../models/safetyProbs/noTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "/wl1_" + str(wl1) + "_wl2_" + str(wl2) + "/"
                    prism_model = "../../models/noTrimming/upperLimit90/if13.5_of4.3_deltawl" + str(deltawl) + "/waterTankBaseline.prism"
                os.makedirs(dir_to_save,exist_ok=True)
                file_to_save = dir_to_save + "/violationProb.txt"

                props_file = "../../models/tankPropsSink.props"


                with open(file_to_save,'w') as f:
                    process = subprocess.Popen(["nice", "-n 10", "/data2/mcleav/safetyContracts/ICCPS_2022/prism/prism-4.5/prism/bin/prism", prism_model, props_file, "-const", "wlidInit1=" + str(wl1) + ",wlidInit2=" + str(wl2) + ",initContAction1=" + str(contAction1) + ",initContAction2=" + str(contAction2), "-javamaxmem", "16g"],stdout=f)

                # out, err = process.communicate()
                # print(out)

                # with open(file_to_save,'w') as file:
                #     file.write(out.decode("utf-8"))


                # if wl1 >= 40 and wl1 <= 70 and wl2>=40 and contAction1==1 and contAction2==1:
                #     counter = min(counter+5,numProc-1)
                # else:
                    # counter +=1

                counter += 1
                sleepTime = 120
                if counter % numProc == numProc-1:
                    counter = 0
                    print("Sleeping for " + str(sleepTime) + " seconds")
                    time.sleep(sleepTime)
                    print("Done sleeping")

