
import numpy as np
import matplotlib.pyplot as plt



# safety params
CAR_POS_THRESH_FOLLOWING = 20 # m
CAR_VEL_LOWER_BOUND_CAR = 4 # m/s
CAR_POS_FOLLOWING_LOWER_BOUND = 14 # m
CAR_POS_FOLLOWING_UPPER_BOUND = 10000#30 # m
CAR_VEL_LOWER_BOUND_NOTHING = 3 # m/s
CAR_VEL_UPPER_BOUND_NOTHING = 3 # m/s

# controller params
GOAL_CAR_SPEED = 10 # m/s
GOAL_CAR_SPEED_THRESH = 1 # m/s

ACCEL_RATE = 4 # m/s^2
BRAKE_RATE = -4 # m/s^2


# Obstacles we can handle
obsts_str = ["nothing","car"]


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
        if obst_type == "car":
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
        if obst_type == "car":
            self.obst_vel = obst_vel
        elif obst_type == "nothing":
            self.obst_vel = 0
        else:
            raise Exception("Unrecognized obstacle type: " + str(obst_type))


    def step(self, obj_class, obj_dist, obj_vel, return_command = False):
        self.cur_step += 1

        # print([self.car_dist,self.car_vel,obj_class])


        ## Compute control command
        if obj_class == "car":
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

        # no rounding
        # self.car_dist = (self.car_dist - (self.car_vel-self.obst_vel)*self.time_step)/self.dist_disc*self.dist_disc
        # self.car_vel = (self.car_vel + command*self.time_step)/self.vel_disc*self.vel_disc

        collision = 0        
        if self.obst_type == "nothing" and self.car_vel <= CAR_VEL_LOWER_BOUND_NOTHING:
            collision = 1
        
        if self.obst_type == "car" and (self.car_vel <= CAR_VEL_LOWER_BOUND_CAR or self.car_dist <= CAR_POS_FOLLOWING_LOWER_BOUND):
            collision = 1
            # print("Collision: dist " + str(self.car_dist) + ", vel " + str(self.car_vel))


        terminal = 0
        if collision:
            terminal = 1

        self.allDist.append(self.car_dist)
        self.allVel.append(self.car_vel)
        
        if return_command:
            return self.car_dist, self.car_vel, collision, terminal, command
        else:
            return self.car_dist, self.car_vel, collision, terminal


    def step_predVel(self, obj_class, obj_dist, obj_vel, return_command = False):
        self.cur_step += 1

        # print([self.car_dist,self.car_vel,obj_class])


        ## Compute control command
        if obj_class == "car":
            command = self.car_following_controller_pred_vel(obj_dist, obj_vel)
        elif obj_class == "nothing":
            command = self.car_vel_controller_pred_vel(obj_vel)
        else:
            raise Exception("Unrecognized obstacle type: " + str(obj_class))


        ## Apply control commands to car model
        # continuous vars
        # self.car_dist = self.car_dist - (self.car_vel-self.obst_vel)*self.time_step
        # self.car_vel = self.car_vel + command*self.time_step

        # discrete vars
        # self.car_dist = round((self.car_dist - (self.car_vel-self.obst_vel)*self.time_step)/self.dist_disc)*self.dist_disc
        # self.car_vel = round((self.car_vel + command*self.time_step)/self.vel_disc)*self.vel_disc

        # no rounding
        self.car_dist = (self.car_dist - (self.car_vel-self.obst_vel)*self.time_step)/self.dist_disc*self.dist_disc
        self.car_vel = (self.car_vel + command*self.time_step)/self.vel_disc*self.vel_disc

        collision = 0
        
        if self.obst_type == "nothing" and (self.car_vel <= GOAL_CAR_SPEED-CAR_VEL_LOWER_BOUND_NOTHING or self.car_vel >= GOAL_CAR_SPEED+CAR_VEL_UPPER_BOUND_NOTHING):
            collision = 1
        
        if self.obst_type == "car" and (self.car_vel <= CAR_VEL_LOWER_BOUND_CAR or self.car_dist <= CAR_POS_FOLLOWING_LOWER_BOUND or self.car_dist >= CAR_POS_FOLLOWING_UPPER_BOUND):
            collision = 1
            # print("Collision: dist " + str(self.car_dist) + ", vel " + str(self.car_vel))


        terminal = 0
        if collision:
            terminal = 1

        self.allDist.append(self.car_dist)
        self.allVel.append(self.car_vel)
        
        if return_command:
            return self.car_dist, self.car_vel, collision, terminal, command
        else:
            return self.car_dist, self.car_vel, collision, terminal


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
        if self.car_vel < GOAL_CAR_SPEED-GOAL_CAR_SPEED_THRESH:
            command = ACCEL_RATE
        elif self.car_vel > GOAL_CAR_SPEED+GOAL_CAR_SPEED_THRESH:
            command = BRAKE_RATE
        return command



    def car_following_controller_pred_vel(self, obj_dist, obj_vel):

        if obj_vel >= 0 and obj_dist > CAR_POS_THRESH_FOLLOWING:
            command = 0
        elif obj_vel >= 0 and obj_dist <= CAR_POS_THRESH_FOLLOWING:
            command = BRAKE_RATE
        elif obj_vel < 0 and obj_dist > CAR_POS_THRESH_FOLLOWING:
            command = ACCEL_RATE
        else:
            command = 0

        return command


    def car_vel_controller_pred_vel(self, car_vel):
        command = 0
        if self.car_vel < GOAL_CAR_SPEED-GOAL_CAR_SPEED_THRESH:
            command = ACCEL_RATE
        elif self.car_vel > GOAL_CAR_SPEED+GOAL_CAR_SPEED_THRESH:
            command = BRAKE_RATE
        return command

