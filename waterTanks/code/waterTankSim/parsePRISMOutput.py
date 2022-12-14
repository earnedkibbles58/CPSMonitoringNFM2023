
import numpy as np
import matplotlib.pyplot as plt
import os

def computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,base_dir):

    if wlid2>wlid1:
        temp = wlid1
        wlid1 = wlid2
        wlid2=temp


    file_name = base_dir + "contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + "/wl1_" + str(wlid1) + "_wl2_" + str(wlid2) + "/violationProb.txt"

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
            line = line.split(" ")
            # print(line)
            temp_line = line[1]
            # print("Temp line")
            # print(temp_line)


            try:
                violation_prob = eval(temp_line)
                # print("Violation prob: " + str(violation_prob))
                # print(violation_prob)

                return violation_prob
            except Exception as e:
                print("Error in evaluation")
                return -1
    
    print("Couldn't find result for " + file_name)
    return -1




def main():

    wlMax = 100
    deltawl = 2
    wlidMax = int(wlMax/deltawl)+1
    trimming = True

    if trimming:
        base_dir = "../../models/safetyProbs/withTrimming/if13.5_of4.3_deltawl" + str(deltawl) + "/"
    else:
        base_dir = "../../models/safetyProbs/noTrimming/if13.5_of4.3_deltawl" + str(deltawl) + "/"
    wlids = range(1,wlidMax)

    
    for contAction1 in range(0,2):
        for contAction2 in range(0,2):

            safeProbs = []
            for wlid1 in range(1,wlidMax):
                tempSafeProbs = []
                for wlid2 in range(1,wlidMax):
                    tempSafeProb = 1-computeViolationFromWaterLevels(wlid1,wlid2,contAction1,contAction2,base_dir)
                    tempSafeProbs.append(tempSafeProb)

                    # print(str([wlid1,wlid2]) + " safe prob: " + str(tempSafeProb))
                    # input("Wait for input")

                safeProbs.append(tempSafeProbs)

            WLIDSY,WLIDSX = np.meshgrid(wlids,wlids)


            if trimming:
                plots_save_dir = "../../results/safetyProbPlots/withTrimming/deltawl_" + str(deltawl) + "/"
            else:
                plots_save_dir = "../../results/safetyProbPlots/noTrimming/deltawl_" + str(deltawl) + "/"
            os.makedirs(plots_save_dir,exist_ok=True)


            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')



            # Plot the values
            ax.scatter(WLIDSX, WLIDSY, safeProbs, c = 'b', marker='o')
            ax.set_xlabel('Water level 1')
            ax.set_ylabel('Water level 2')
            ax.set_zlabel('Safety Probability')

            plt.savefig(plots_save_dir + "safeProbs_contAction1_" + str(contAction1) + "_contAction2_" + str(contAction2) + ".png")

if __name__ == '__main__':
    main()
