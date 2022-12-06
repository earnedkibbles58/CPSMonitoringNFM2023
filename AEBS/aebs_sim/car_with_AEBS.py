
import numpy as np
import matplotlib.pyplot as plt



# safety params
CAR_POS_THRESH_FOLLOWING = 30 # m
CAR_POS_THRESH_AEBS = 5 # m
CAR_VEL_LOWER_BOUND = 4 # m/s
CAR_POS_FOLLOWING_LOWER_BOUND = CAR_POS_THRESH_FOLLOWING/4 # m


# controller params
OTHER_CAR_SPEED = 8 # m/s
GOAL_CAR_SPEED = 10 # m/s

ACCEL_RATE = 4 # m/s^2
BRAKE_RATE = -4 # m/s^2

B1 = 4 # m/s^2
B2 = 8 # m/s^2
TTC_THRESH = 6 # s
X_WARNING = 1 # idk the units
F_MU = 1 # coefficient of friction
T_HD = 2 # s
T_S_DELAY = 0 # s

# Obstacles we can handle
obsts_str = ["nothing","rock","car"]


class World:

    def __init__(self, car_dist, car_vel, obst_type, obst_vel, dist_disc, vel_disc, \
                 episode_length, time_step):

        # car states
        self.car_dist = car_dist
        self.car_vel = car_vel
        self.goal_car_vel = self.car_vel

        self.dist_disc = dist_disc
        self.vel_disc = vel_disc

        

        # step parameters
        self.time_step = time_step
        self.cur_step = 0
        self.episode_length = episode_length

        # storage
        self.allDist = []
        self.allVel = []
        self.allDist.append(self.car_dist)
        self.allVel.append(self.car_vel)

        self.obst_type = obst_type
        if obst_type == "rock":
            self.obst_vel =  0
        elif obst_type == "car":
            self.obst_vel = obst_vel
        elif obst_type == "nothing":
            self.obst_vel = 0
        else:
            raise Exception("Unrecognized obstacle type: " + str(obst_type))




    def reset(self, car_dist, car_vel, obst_type, obst_vel, seed=None):
        if not seed == None:
            np.random.seed(seed)
        # car states
        self.car_dist = car_dist
        self.car_vel = car_vel

        # step parameters
        self.cur_step = 0

        # storage
        self.allDist = []
        self.allVel = []
        self.allDist.append(self.car_dist)
        self.allVel.append(self.car_vel)

        self.obst_type = obst_type
        if obst_type == "rock":
            self.obst_vel =  0
        elif obst_type == "car":
            self.obst_vel = obst_vel
        elif obst_type == "nothing":
            self.obst_vel = 0
        else:
            raise Exception("Unrecognized obstacle type: " + str(obst_type))


    def step(self, obj_class, obj_dist, obj_vel, return_command = False):
        self.cur_step += 1

        # print([self.car_dist,self.car_vel,obj_class])


        ## Compute control command
        if obj_class == "rock":
            command = self.AEBS_controller(obj_dist, obj_vel)
        elif obj_class == "car":
            command = self.car_following_controller(obj_dist, obj_vel)
        elif obj_class == "nothing":
            command = self.car_vel_controller(obj_dist, obj_vel)
        else:
            raise Exception("Unrecognized obstacle type: " + str(obj_class))


        ## Apply control commands to car model
        # continuous vars
        # self.car_dist = self.car_dist - (self.car_vel-self.obst_vel)*self.time_step
        # self.car_vel = self.car_vel + command*self.time_step

        # discrete vars
        self.car_dist = round((self.car_dist - (self.car_vel-self.obst_vel)*self.time_step)/self.dist_disc)*self.dist_disc
        self.car_vel = round((self.car_vel + command*self.time_step)/self.vel_disc)*self.vel_disc

        collision = 0
        if self.obst_type == "rock" and self.car_dist <= CAR_POS_THRESH_AEBS:
            collision = 1
        
        if self.obst_type == "nothing" and self.car_vel <= CAR_VEL_LOWER_BOUND:
            collision = 1
        
        if self.obst_type == "car" and (self.car_vel <= CAR_VEL_LOWER_BOUND or self.car_dist <= CAR_POS_FOLLOWING_LOWER_BOUND):
            collision = 1
            # print("Collision: dist " + str(self.car_dist) + ", vel " + str(self.car_vel))


        terminal = 0
        if collision or self.car_vel <= 0:
            terminal = 1

        self.allDist.append(self.car_dist)
        self.allVel.append(self.car_vel)
        
        if return_command:
            return self.car_dist, self.car_vel, collision, terminal, command
        else:
            return self.car_dist, self.car_vel, collision, terminal

    def AEBS_controller(self,obj_dist, obj_vel):

        dist = obj_dist-CAR_POS_THRESH_AEBS
        if(self.car_vel==0):
            return 0
        dbr = self.car_vel*T_S_DELAY + F_MU*((self.car_vel**2)/(2*B2))
        dw = self.car_vel*T_S_DELAY + F_MU*((self.car_vel**2)/(2*B2)) + self.car_vel*T_HD
        TTC = dist/self.car_vel
        x = (dist-dbr)/(dw-dbr)

        if(TTC<=TTC_THRESH and x <= X_WARNING):
            command = B2
        elif TTC<=TTC_THRESH:
            command = B1
        elif x <= X_WARNING:
            command = B1
        else:
            command = 0
        return -command


    def car_following_controller(self, obj_dist, obj_vel):

        if self.car_vel >= obj_vel and obj_dist > CAR_POS_THRESH_FOLLOWING:
            command = 0
        elif self.car_vel >= obj_vel and obj_dist <= CAR_POS_THRESH_FOLLOWING:
            command = BRAKE_RATE
        elif self.car_vel < obj_vel and obj_dist > CAR_POS_THRESH_FOLLOWING:
            command = ACCEL_RATE
        else:
            command = 0

        return command


    def car_vel_controller(self, obj_dist, obj_vel):
        command = 0
        if self.car_vel < self.goal_car_vel:
            command = ACCEL_RATE
        return command