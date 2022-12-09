
import math
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from statistics import mode, mean, _counts

PARTICLE_INIT_DIST = 30
PARTICLE_INIT_SPEED = 10
PARTICLE_DIST_NOISE = 10
PARTICLE_SPEED_NOISE = 5

NOTHING_DIST = -100

PARTICLE_DYN_DIST_NOISE = 1
PARTICLE_DYN_VEL_NOISE = 1
PARTICLE_DYN_OBST_VEL_NOISE = 1

class particleFilter:

    def __init__(self, car_speed, classifier_accs, time_step, dist_disc, vel_disc, num_particles = 1000, max_dist=None,max_vel=None) -> None:

        self.num_particles = num_particles
        self.classifier_accs = classifier_accs
        self.time_step = time_step

        self.dist_disc = dist_disc
        self.vel_disc = vel_disc

        self.allParticles = []
        particle_list = []

        self.max_dist = max_dist
        self.max_vel = max_vel

        for i in range(self.num_particles):
            init_obst = random.choice([0,1,2])

            init_dist = NOTHING_DIST if init_obst == 0 else PARTICLE_INIT_DIST + PARTICLE_DIST_NOISE * (2*np.random.random() - 1)
            init_speed_obst = 0 if init_obst != 2 else PARTICLE_INIT_SPEED
            init_speed = car_speed - init_speed_obst + PARTICLE_SPEED_NOISE * (2*np.random.random() - 1)
            
            
            particle_list.append([init_dist,init_speed,init_obst]) # distance, speed, obstacle class
        
        self.allParticles.append(particle_list)



    def resample_particles(self, weights):

        new_indices = np.random.choice(self.num_particles, self.num_particles, p=weights)

        particle_list = []
        for i in range(len(new_indices)):

            new_particle = copy.deepcopy(self.allParticles[len(self.allParticles)-1][new_indices[i]])
            particle_list.append(new_particle)

        self.allParticles.append(particle_list)



    def compute_weights(self,est_obst,est_dist,est_vel,car_vel):
        weights = np.zeros(self.num_particles)
        for i in range(self.num_particles):

            curr_particle = self.allParticles[len(self.allParticles)-1][i]
            if est_obst == 0 or curr_particle[2] == 0:
                weight = self.classifier_accs[curr_particle[2]][est_obst]*(np.random.random()/2)
            else:
                # weight = math.exp(-abs(curr_particle[0]-est_dist)/2 - abs(curr_particle[1]-(car_vel-est_vel))/2)*self.classifier_accs[curr_particle[2]][est_obst] # with vel error in weight
                weight = math.exp(-abs(curr_particle[0]-est_dist)/2)*self.classifier_accs[curr_particle[2]][est_obst] # without vel error in weight
            weights[i] = weight

        # normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

            # make sure there are no precision issues
            assert(np.abs(np.sum(weights) - 1) < 1e-6)
        else:
            weights = [1/self.num_particles] * self.num_particles

        return weights


    def step_dynamics(self, control_command):
        for i in range(self.num_particles):

            curr_particle = self.allParticles[len(self.allParticles)-1][i]

            # rounding
            # if self.max_dist is not None:
            #     curr_particle[0] = NOTHING_DIST if curr_particle[2] == 0 else round((min(self.max_dist,curr_particle[0] - (curr_particle[1])*self.time_step) + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0])/self.dist_disc)*self.dist_disc# + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0]
            # else:
            #     curr_particle[0] = NOTHING_DIST if curr_particle[2] == 0 else round((curr_particle[0] - (curr_particle[1])*self.time_step + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0])/self.dist_disc)*self.dist_disc# + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0]
            # if self.max_vel is not None:
            #     curr_particle[1] = max(-self.max_vel,round(min(self.max_vel,(curr_particle[1] + control_command*self.time_step + np.random.normal(0,PARTICLE_DYN_VEL_NOISE,1)[0]))/self.vel_disc)*self.vel_disc)
            # else:
            #     curr_particle[1] = round((curr_particle[1] + control_command*self.time_step + np.random.normal(0,PARTICLE_DYN_VEL_NOISE,1)[0])/self.vel_disc)*self.vel_disc

            # no rounding
            if self.max_dist is not None:
                curr_particle[0] = NOTHING_DIST if curr_particle[2] == 0 else (min(self.max_dist,curr_particle[0] - (curr_particle[1])*self.time_step) + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0])/self.dist_disc*self.dist_disc# + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0]
            else:
                curr_particle[0] = NOTHING_DIST if curr_particle[2] == 0 else (curr_particle[0] - (curr_particle[1])*self.time_step + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0])/self.dist_disc*self.dist_disc# + np.random.normal(0,PARTICLE_DYN_DIST_NOISE,1)[0]
            if self.max_vel is not None:
                curr_particle[1] = max(-self.max_vel,min(self.max_vel,(curr_particle[1] + control_command*self.time_step + np.random.normal(0,PARTICLE_DYN_VEL_NOISE,1)[0]))/self.vel_disc*self.vel_disc)
            else:
                curr_particle[1] = (curr_particle[1] + control_command*self.time_step + np.random.normal(0,PARTICLE_DYN_VEL_NOISE,1)[0])/self.vel_disc*self.vel_disc

            self.allParticles[len(self.allParticles)-1][i] = curr_particle


    def step_filter(self,est_obst,est_dist,est_vel,car_vel,control_command):
        # takes in the sensor readings and advances the filter by one step
        self.step_dynamics(control_command)
        weights = self.compute_weights(est_obst,est_dist,est_vel,car_vel)
        self.resample_particles(weights)

        return self.allParticles[len(self.allParticles)-1]

    

    def plot_filter_states(self,folder_to_save):


        for time_step in range(len(self.allParticles)):

            particle_dists = [self.allParticles[time_step][i][0] for i in range(self.num_particles)]
            particle_vels = [self.allParticles[time_step][i][1] for i in range(self.num_particles)]
            particle_obst_classes = [self.allParticles[time_step][i][2] for i in range(self.num_particles)]

            ## Plot 2-d dots of distances and speeds
            plt.plot(particle_dists,particle_vels,'b.')
            plt.xlabel("Distance from car to obstacle")
            plt.ylabel("Speed of car")
            plt.savefig(folder_to_save + "/carStates/carStatesPF" + str(time_step) + ".png")
            plt.clf()

            ## plot histogram of obst classes
            plt.hist(particle_obst_classes,bins=3)
            plt.savefig(folder_to_save + "/obstClasses/obstClassesPF" + str(time_step) + ".png")
            plt.clf()


    def get_filter_state(self):
        all_obst_class = [particle[2] for particle in self.allParticles[len(self.allParticles)-1]]
        pred_obst = max([p[0] for p in _counts(all_obst_class)])
        # pred_obst = mode(all_obst_class)

        all_dist = []
        all_vel = []
        for i in range(self.num_particles):

            curr_particle = self.allParticles[len(self.allParticles)-1][i]
            if curr_particle[2] != pred_obst:
                continue
            all_dist.append(curr_particle[0])
            all_vel.append(curr_particle[1])

        pred_dist = mean(all_dist)
        pred_vel = mean(all_vel)

        if pred_obst==0:
            assert pred_dist == NOTHING_DIST

        return pred_dist,pred_vel,pred_obst