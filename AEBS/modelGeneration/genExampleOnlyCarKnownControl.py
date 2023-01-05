import numpy as np
import csv
import math

# distDiscs = [0.01,0.05,0.1,0.5,1,1.5,2]


obsts = [0,1] # nothing, rock, car
obsts_str = ["nothing","car"]


distDiscs = [0.5]
velDisc = 0.4

initDist = 120
initVel = 20

max_time=30

controlActions = [0, 4, -4, -8]


## controller params
# no obstacle
GOAL_CAR_SPEED = 10 # m/s
GOAL_CAR_SPEED_THRESH = 0 # m/s

# car obstacle
CAR_POS_THRESH_FOLLOWING = 20 # m



## no obst perception errors (given as indices assuming 0.4 velocity discretization)
noObstDetCounts = [53759, 517]
noObstDetProbs = [noObstDetCounts[0]/sum(noObstDetCounts), noObstDetCounts[1]/sum(noObstDetCounts)]

## no obst det
# noObstVelErrs = {1: 0.26888495836099696, -2: 0.05269363991451256, 0: 0.27240400913850493, 2: 0.1616183948706755, 3: 0.05534674625985889, 4: 0.010999336723413654, -1: 0.15542781339820075, -3: 0.010372908836318053, -17: 0.0010317635787456707, -20: 0.0012897044734320885, -23: 0.0007001252855774195, 5: 0.0014370992703957558, -19: 0.0009396418306433788, -16: 0.0006817009359569611, -4: 0.0010501879283661291, -21: 0.0010317635787456707, -18: 0.000902793131402462, -15: 0.0003500626427887097, -28: 9.212174810229198e-05, -27: 0.00014739479696366717, -24: 0.0005343061389932938, -30: 3.684869924091679e-05, -26: 0.00025794089468641763, -25: 0.0004053356916500849, -14: 0.0002026678458250424, -22: 0.0007369739848183363, 6: 9.212174810229198e-05, -11: 1.8424349620458396e-05, 8: 1.8424349620458396e-05, 7: 3.684869924091679e-05, -13: 9.212174810229198e-05, -5: 9.212174810229198e-05, -29: 1.8424349620458396e-05, -12: 5.527304886137519e-05}
noObstNoObstDetVelErrs = {1: 0.27147082349005536, -2: 0.0532003943525698, 0: 0.2750237169590566, 2: 0.16317267806322214, 3: 0.05587901560668491, 4: 0.011105117282687614, -1: 0.15692256180361788, -3: 0.01047266504213254, 5: 0.0014509198459792768, -4: 0.0010602875797540883, 6: 9.30076824345691e-05, 8: 1.860153648691382e-05, 7: 3.720307297382764e-05, -5: 9.30076824345691e-05}

## car obst det
noObstCarDetDistErrs = {3: 0.06062777698150673, 1: 0.2957395728824348, 0: 0.2646911279920595, -1: 0.12209760642108843, -3: 0.00507023075820555, -2: 0.03103052888060642, 2: 0.1634119248961037, 7: 0.00444317041708472, 5: 0.01066002579905409, 4: 0.02218002006593042, 11: 0.0010032965457933214, 8: 0.00286656155940949, 9: 0.0021678371793034225, 6: 0.006915579762075421, 10: 0.0017557689551383083, -5: 0.00041206822416511407, -4: 0.0008420524580765377, 12: 0.0009495485165543936, 17: 0.00023290812670202088, 15: 0.0004479002436577327, 13: 0.0007166403898523724, 16: 0.0003583201949261861, 14: 0.000465816253404042, 23: 5.37480292389279e-05, 22: 0.00012541206822416513, 18: 0.00010749605847785582, 21: 0.00014332807797047442, -7: 8.958004873154651e-05, 20: 8.958004873154651e-05, -6: 7.166403898523721e-05, 24: 5.37480292389279e-05, 19: 7.166403898523721e-05, 26: 1.7916009746309302e-05, 25: 3.5832019492618604e-05, 31: 1.7916009746309302e-05, 30: 1.7916009746309302e-05, 27: 1.7916009746309302e-05}
noObstCarDetVelErrs = {-2: 0.09289451053460689, -1: 0.14030027232334888, 4: 0.05396302135588027, 1: 0.1724595098179961, 3: 0.09719435287372073, 5: 0.02472409344990611, 2: 0.14275476565859493, 0: 0.17407195069516504, 6: 0.009172996990110408, -3: 0.049484018919303346, -5: 0.009620897233768144, -6: 0.002812813530170562, -4: 0.023165400601977342, -8: 0.0007883044288376097, 7: 0.003368209832306154, 8: 0.0010032965457933214, -7: 0.0012362046724953412, -9: 0.0001612440877167837, 9: 0.0003404041851798768, -10: 0.00012541206822416513, -14: 3.5832019492618604e-05, -12: 8.958004873154651e-05, 10: 0.00010749605847785582, -11: 5.37480292389279e-05, 13: 1.7916009746309302e-05, 11: 1.7916009746309302e-05, -13: 1.7916009746309302e-05, 12: 1.7916009746309302e-05}


## car obst perception errors (given as indices assuming 0.5 distance and 0.4 velocity discretiation)
carObstDetCounts = [2928, 55816]
carObstDetProbs = [carObstDetCounts[0]/sum(carObstDetCounts), carObstDetCounts[1]/sum(carObstDetCounts)]


# no obst
carObstNoObstDetErrs = {1: 0.27147082349005536, -2: 0.0532003943525698, 0: 0.2750237169590566, 2: 0.16317267806322214, 3: 0.05587901560668491, 4: 0.011105117282687614, -1: 0.15692256180361788, -3: 0.01047266504213254, 5: 0.0014509198459792768, -4: 0.0010602875797540883, 6: 9.30076824345691e-05, 8: 1.860153648691382e-05, 7: 3.720307297382764e-05, -5: 9.30076824345691e-05}

# car obst
carObstCarDetDistErrs = {3: 0.06062777698150673, 1: 0.2957395728824348, 0: 0.2646911279920595, -1: 0.12209760642108843, -3: 0.00507023075820555, -2: 0.03103052888060642, 2: 0.1634119248961037, 7: 0.00444317041708472, 5: 0.01066002579905409, 4: 0.02218002006593042, 11: 0.0010032965457933214, 8: 0.00286656155940949, 9: 0.0021678371793034225, 6: 0.006915579762075421, 10: 0.0017557689551383083, -5: 0.00041206822416511407, -4: 0.0008420524580765377, 12: 0.0009495485165543936, 17: 0.00023290812670202088, 15: 0.0004479002436577327, 13: 0.0007166403898523724, 16: 0.0003583201949261861, 14: 0.000465816253404042, 23: 5.37480292389279e-05, 22: 0.00012541206822416513, 18: 0.00010749605847785582, 21: 0.00014332807797047442, -7: 8.958004873154651e-05, 20: 8.958004873154651e-05, -6: 7.166403898523721e-05, 24: 5.37480292389279e-05, 19: 7.166403898523721e-05, 26: 1.7916009746309302e-05, 25: 3.5832019492618604e-05, 31: 1.7916009746309302e-05, 30: 1.7916009746309302e-05, 27: 1.7916009746309302e-05}
carObstCarDetVelErrs = {-2: 0.09289451053460689, -1: 0.14030027232334888, 4: 0.05396302135588027, 1: 0.1724595098179961, 3: 0.09719435287372073, 5: 0.02472409344990611, 2: 0.14275476565859493, 0: 0.17407195069516504, 6: 0.009172996990110408, -3: 0.049484018919303346, -5: 0.009620897233768144, -6: 0.002812813530170562, -4: 0.023165400601977342, -8: 0.0007883044288376097, 7: 0.003368209832306154, 8: 0.0010032965457933214, -7: 0.0012362046724953412, -9: 0.0001612440877167837, 9: 0.0003404041851798768, -10: 0.00012541206822416513, -14: 3.5832019492618604e-05, -12: 8.958004873154651e-05, 10: 0.00010749605847785582, -11: 5.37480292389279e-05, 13: 1.7916009746309302e-05, 11: 1.7916009746309302e-05, -13: 1.7916009746309302e-05, 12: 1.7916009746309302e-05}

# carObstDistErrs = {3: 0.06062777698150673, 1: 0.2957395728824348, 0: 0.2646911279920595, -1: 0.12209760642108843, -3: 0.00507023075820555, -2: 0.03103052888060642, 2: 0.1634119248961037, 7: 0.00444317041708472, 5: 0.01066002579905409, 4: 0.02218002006593042, 11: 0.0010032965457933214, 8: 0.00286656155940949, 9: 0.0021678371793034225, 6: 0.006915579762075421, 10: 0.0017557689551383083, -5: 0.00041206822416511407, -4: 0.0008420524580765377, 12: 0.0009495485165543936, 17: 0.00023290812670202088, 15: 0.0004479002436577327, 13: 0.0007166403898523724, 16: 0.0003583201949261861, 14: 0.000465816253404042, 23: 5.37480292389279e-05, 22: 0.00012541206822416513, 18: 0.00010749605847785582, 21: 0.00014332807797047442, -7: 8.958004873154651e-05, 20: 8.958004873154651e-05, -6: 7.166403898523721e-05, 24: 5.37480292389279e-05, 19: 7.166403898523721e-05, 26: 1.7916009746309302e-05, 25: 3.5832019492618604e-05, 31: 1.7916009746309302e-05, 30: 1.7916009746309302e-05, 27: 1.7916009746309302e-05}
# carObstVelErrs = {-2: 0.09289451053460689, -1: 0.14030027232334888, 4: 0.05396302135588027, 1: 0.1724595098179961, 3: 0.09719435287372073, 5: 0.02472409344990611, 2: 0.14275476565859493, 0: 0.17407195069516504, 6: 0.009172996990110408, -3: 0.049484018919303346, -5: 0.009620897233768144, -6: 0.002812813530170562, -4: 0.023165400601977342, -8: 0.0007883044288376097, 7: 0.003368209832306154, 8: 0.0010032965457933214, -7: 0.0012362046724953412, -9: 0.0001612440877167837, 9: 0.0003404041851798768, -10: 0.00012541206822416513, -14: 3.5832019492618604e-05, -12: 8.958004873154651e-05, 10: 0.00010749605847785582, -11: 5.37480292389279e-05, 13: 1.7916009746309302e-05, 11: 1.7916009746309302e-05, -13: 1.7916009746309302e-05, 12: 1.7916009746309302e-05}

for i,obst in enumerate(obsts):

    for distDisc in distDiscs:
        speedDisc = 0.4


        # AEBS braking powers
        B1=4
        accelRate = int(B1/speedDisc)
        brakeRate = int(B1/speedDisc)

        modelFile = open("../models/knownControl3/carModel_DistDisc" + str(distDisc) + "_" + obsts_str[i] + ".prism",'w')

        modelFile.write("mdp\n\n")
        
        # AEBS params
        modelFile.write("const int freq = 10;\n")

        # car following params
        otherCarSpeed = 8

        # detection params
        objDistIfNoObst = 10


        modelFile.write("const int carPosThreshFollowing=" + str(int(CAR_POS_THRESH_FOLLOWING/distDisc)) + ";\n")

        modelFile.write("const int initPos;\n")
        modelFile.write("const int initSpeed;\n")

        modelFile.write("const int initCont;\n")
        modelFile.write("const int initState;\n")
        
        
        modelFile.write("const int otherCarSpeed=" + str(int(otherCarSpeed/speedDisc)) + ";\n")
        # modelFile.write("const int goalCarSpeed=" + str(int(goalCarSpeed/speedDisc)) + ";\n\n")

        modelFile.write("const int goalCarSpeedLower=" + str(int((GOAL_CAR_SPEED-GOAL_CAR_SPEED_THRESH)/speedDisc)) + ";\n\n")
        modelFile.write("const int goalCarSpeedUpper=" + str(int((GOAL_CAR_SPEED+GOAL_CAR_SPEED_THRESH)/speedDisc)) + ";\n\n")

        if obst==0:
            modelFile.write("const int obstSpeed=0;\n\n") ## FIXME
        elif obst==1:
            modelFile.write("const int obstSpeed=" + str(int(otherCarSpeed/speedDisc)) + ";\n\n") ## FIXME
        else:
            print("Unknown obstacle type. Quitting...")
            modelFile.close()
            break



        modelFile.write("module LECMarkovChain\n\n")

        modelFile.write("    carSpeed : [0..initSpeed+100] init initSpeed;\n")
        modelFile.write("    carPos : [0..initPos+100] init initPos;\n\n")
        modelFile.write("    carSpeedDet : [-initSpeed-100..initSpeed+200] init initSpeed;\n\n")
        modelFile.write("    carPosDet : [-initPos-100..initPos+200] init initPos;\n\n")


        modelFile.write("    seqflag : [0..2] init initState;\n") ## FIXME: set back to init initBrakingFlag

        modelFile.write("    perFlag : [0..2] init 0;\n\n") # 0 is class reading, 1 is vel reading, 2 is distance reading


        modelFile.write("    lecDet : [0..1] init 0;\n\n") # 0 is none, 1 is car
        modelFile.write("    accelCommand : [-100..100] init initCont;\n\n") # negative value is braking, positive value is accelerating

        modelFile.write("    time : [0.." + str(max_time) + "] init " + str(max_time) + ";\n\n") ## FIXME: was 15

        # lec model
        modelFile.write("    // generate detection output\n")


        if obst == 0:
            modelFile.write("    [] seqflag=0&perFlag=0 -> " + str(noObstDetProbs[0]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=0) + " + str(noObstDetProbs[1]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=1);\n")

        else:
            modelFile.write("    [] seqflag=0&perFlag=0 -> " + str(carObstDetProbs[0]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=0) + " + str(carObstDetProbs[1]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=1);\n")


        if obst == 0:

            # modelFile.write("    [] seqflag=0&perFlag=0 -> " + str(noObstDetProbs[0]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=0) + " + str(noObstDetProbs[1]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=1);\n")

            # if no obst det

            firstIterFlag = True
            for key in noObstNoObstDetVelErrs:
                transProb = noObstNoObstDetVelErrs[key]
                if firstIterFlag:
                    firstIterFlag = False
                    modelFile.write("    [] seqflag=0&perFlag=1&lecDet=0 -> " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(goalCarSpeedUpper+1,max(goalCarSpeedLower-1,carSpeed+" + str(int(key)) + ")))")
                else:
                    modelFile.write(" + " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(goalCarSpeedUpper+1,max(goalCarSpeedLower-1,carSpeed+" + str(int(key)) + ")))")
            modelFile.write(";\n")





            # if car det

            firstIterFlag = True
            for key in noObstCarDetDistErrs:
                transProb = noObstCarDetDistErrs[key]
                if firstIterFlag:
                    firstIterFlag = False
                    modelFile.write("    [] seqflag=0&perFlag=1&lecDet=1 -> " + str(transProb) + ":(seqflag'=0)&(perFlag'=2)&(carPosDet'=min(carPosThreshFollowing+1,max(carPosThreshFollowing-1,carPos+" + str(int(key)) + ")))")
                else:
                    modelFile.write(" + " + str(transProb) + ":(seqflag'=0)&(perFlag'=2)&(carPosDet'=min(carPosThreshFollowing+1,max(carPosThreshFollowing-1,carPos+" + str(int(key)) + ")))")
            modelFile.write(";\n")


            firstIterFlag = True
            runningSum = 0
            for key in noObstCarDetVelErrs:
                transProb = noObstCarDetVelErrs[key]
                runningSum+=transProb
                if firstIterFlag:
                    firstIterFlag = False
                    modelFile.write("    [] seqflag=0&perFlag=2&lecDet=1 -> " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(otherCarSpeed+1,max(otherCarSpeed-1,carSpeed+" + str(int(key)) + ")))")
                else:
                    modelFile.write(" + " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(otherCarSpeed+1,max(otherCarSpeed-1,carSpeed+" + str(int(key)) + ")))")
            modelFile.write(";\n\n\n")
            print(runningSum)




        
        elif obst == 1:

            # modelFile.write("    [] seqflag=0&perFlag=0 -> " + str(carObstDetProbs[0]) + ":(seqflag'=0)&(perFlag'=2)&(lecDet'=0) + " + str(carObstDetProbs[1]) + ":(seqflag'=0)&(perFlag'=1)&(lecDet'=1);\n")

            # no det
            firstIterFlag = True
            for key in carObstNoObstDetErrs:
                transProb = carObstNoObstDetErrs[key]
                if firstIterFlag:
                    firstIterFlag = False
                    modelFile.write("    [] seqflag=0&perFlag=1&lecDet=0 -> " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(goalCarSpeedUpper+1,max(goalCarSpeedLower-1,carSpeed+" + str(int(key)) + ")))")
                else:
                    modelFile.write(" + " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(goalCarSpeedUpper+1,max(goalCarSpeedLower-1,carSpeed+" + str(int(key)) + ")))")
            modelFile.write(";\n")



            # car det
            firstIterFlag = True
            for key in carObstCarDetDistErrs:
                transProb = carObstCarDetDistErrs[key]
                if firstIterFlag:
                    firstIterFlag = False
                    modelFile.write("    [] seqflag=0&perFlag=1&lecDet=1 -> " + str(transProb) + ":(seqflag'=0)&(perFlag'=2)&(carPosDet'=min(carPosThreshFollowing+1,max(carPosThreshFollowing-1,carPos+" + str(int(key)) + ")))")
                else:
                    modelFile.write(" + " + str(transProb) + ":(seqflag'=0)&(perFlag'=2)&(carPosDet'=min(carPosThreshFollowing+1,max(carPosThreshFollowing-1,carPos+" + str(int(key)) + ")))")
            modelFile.write(";\n")


            firstIterFlag = True
            runningSum = 0
            for key in carObstCarDetVelErrs:
                transProb = carObstCarDetVelErrs[key]
                runningSum+=transProb
                if firstIterFlag:
                    firstIterFlag = False
                    modelFile.write("    [] seqflag=0&perFlag=2&lecDet=1 -> " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(otherCarSpeed+1,max(otherCarSpeed-1,carSpeed+" + str(int(key)) + ")))")
                else:
                    modelFile.write(" + " + str(transProb) + ":(seqflag'=1)&(perFlag'=0)&(carSpeedDet'=min(otherCarSpeed+1,max(otherCarSpeed-1,carSpeed+" + str(int(key)) + ")))")
            modelFile.write(";\n\n\n")
            print(runningSum)




        ## no detection controller
        modelFile.write("    [] seqflag=1&lecDet=0&carSpeedDet>goalCarSpeedUpper -> (seqflag'=2)&(accelCommand'=0);\n")
        modelFile.write("    [] seqflag=1&lecDet=0&carSpeedDet<goalCarSpeedLower -> (seqflag'=2)&(accelCommand'=0);\n")
        modelFile.write("    [] seqflag=1&lecDet=0&carSpeedDet>=goalCarSpeedLower&carSpeedDet<=goalCarSpeedUpper -> (seqflag'=2)&(accelCommand'=" + str(accelRate) + ");\n\n")



        ## car following controller
        modelFile.write("    [] seqflag=1&lecDet=1&carSpeedDet>=otherCarSpeed&carPosDet>carPosThreshFollowing -> (seqflag'=2)&(accelCommand'=0);\n")
        modelFile.write("    [] seqflag=1&lecDet=1&carSpeedDet>=otherCarSpeed&carPosDet<=carPosThreshFollowing -> (seqflag'=2)&(accelCommand'=-" + str(brakeRate) + ");\n")
        modelFile.write("    [] seqflag=1&lecDet=1&carSpeedDet<otherCarSpeed&carPosDet>carPosThreshFollowing -> (seqflag'=2)&(accelCommand'=" + str(accelRate) + ");\n")
        modelFile.write("    [] seqflag=1&lecDet=1&carSpeedDet<otherCarSpeed&carPosDet<=carPosThreshFollowing -> (seqflag'=2)&(accelCommand'=0);\n\n\n")




        ## update car speed and dist based on braking command and obst speed
        modelFile.write("    // compute carPos, carSpeed\n")        
        if obst == 0:
            # no obst
            modelFile.write("    [] seqflag=2&carSpeed>0&time>0 -> (carSpeed'=min(initSpeed+100,max(0,carSpeed+round(accelCommand/freq))))&(seqflag'=0)&(time'=time-1);\n")
        else:
            # car
            modelFile.write("    [] seqflag=2&carSpeed>0&carPos>0&time>0 -> (carPos'=min(initPos,max(0,round(carPos-" + str(speedDisc/distDisc) + "*(carSpeed-obstSpeed)/freq))))&(carSpeed'=min(initSpeed+100,max(0,carSpeed+round(accelCommand/freq))))&(seqflag'=0)&(time'=time-1);\n")

        modelFile.write("endmodule")
        modelFile.close()



