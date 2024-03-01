# BSD 2-Clause License

# Copyright (c) 2023, Bandi Jai Krishna

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import time
import math
import numpy as np
import struct
import torch
import scipy
import csv

from actor_critic import ActorCritic
# from LinearKalmanFilter import LinearKFPositionVelocityEstimator
# from FastLinearKF import FastLinearKFPositionVelocityEstimator

sys.path.append('unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

class Agent():
    def __init__(self,path):
        self.dt = 0.02
        self.num_actions = 12
        self.num_obs = 45*5 #44*5 #48
        self.unit_obs = 45
        self.num_privl_obs =  466 #self.num_obs #421 # num_obs 
        self.device = 'cpu'
        self.path = path#'bp4/model_1750.pt'
        self.d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
                'FL_0':3, 'FL_1':4, 'FL_2':5, 
                'RR_0':6, 'RR_1':7, 'RR_2':8, 
                'RL_0':9, 'RL_1':10, 'RL_2':11 }
        PosStopF  = math.pow(10,9)
        VelStopF  = 16000.0
        HIGHLEVEL = 0xee
        LOWLEVEL  = 0xff
        self.init = True
        self.motiontime = 0 
        self.timestep = 0
        self.time = 0
        self.initialized = False
        self.init_log_data = False

        self.base_ang_vel_list = []
        self.dof_pos_list = []
        self.projected_gravity_list = []
        self.dof_vel_list = []

        self.finish_rec = 0
        self.leg_data_buffer = []
        self.csv_filename = "joint_angles.csv"

#####################################################################
        self.euler = np.zeros(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.omegaBody = np.zeros(3)
        self.accel = np.zeros(3)
        self.smoothing_ratio = 0.2

        self.default_angles = [0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1,-1.5,-0.1,1,-1.5]
        self.default_angles_tensor = torch.tensor([0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1,-1.5,-0.1,1,-1.5],device=self.device,dtype=torch.float,requires_grad=False)
        
        self.actions = torch.zeros(self.num_actions,device=self.device,dtype=torch.float,requires_grad=False)
        self.obs = torch.zeros(self.num_obs,device=self.device,dtype=torch.float,requires_grad=False)
        self.obs_storage = torch.zeros(self.unit_obs*4,device=self.device,dtype=torch.float)


        actor_critic = ActorCritic(num_actor_obs=self.num_obs,num_critic_obs=self.num_privl_obs,num_actions=12,actor_hidden_dims = [512, 256, 128],critic_hidden_dims = [512, 256, 128],activation = 'elu',init_noise_std = 1.0)
        loaded_dict = torch.load(self.path, map_location=torch.device("cpu"))
        actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        actor_critic.eval()
        self.policy = actor_critic.act_inference

        self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)

        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

    def get_observations(self):
        # Commands
        lx = struct.unpack('f', struct.pack('4B', *self.state.wirelessRemote[4:8]))
        ly = struct.unpack('f', struct.pack('4B', *self.state.wirelessRemote[20:24]))
        rx = struct.unpack('f', struct.pack('4B', *self.state.wirelessRemote[8:12]))
        # ry = struct.unpack('f', struct.pack('4B', *self.state.wirelessRemote[12:16]))
        forward = ly[0]*0.6  
        if abs(forward) <0.30:
            forward = 0
        side = -lx[0]*0.5
        if abs(side) <0.2:
            side = 0
        rotate = -rx[0]*0.8
        if abs(rotate) <0.4:
            rotate = 0

        # Get sensor data and compute necessary quantities
        self.q = self.getJointPos()
        self.dq = self.getJointVelocity()
        self.quat = self.getQuaternion()
        self.omegaBody = self.getBodyAngularVel() #self.state.imu.gyroscope
        self.contact_estimate = self.contactEstimator()
        self.aBody = self.getBodyAccel()
        #vel_estimator = FastLinearKFPositionVelocityEstimator(q, dq, p, v, quat, omegaBody, contact_estimate, body_accel)
        # self.base_lin_vel = self.runKF()
        # self.lin_vel = self.aBody*0.001
        self.R = self.get_rotation_matrix_from_rpy(self.state.imu.rpy)
        self.gravity_vector = self.get_gravity_vector()
        self.pitch = torch.tensor([self.state.imu.rpy[1]],device=self.device,dtype=torch.float,requires_grad=False)
        self.roll = torch.tensor([self.state.imu.rpy[0]],device=self.device,dtype=torch.float,requires_grad=False)

        self.base_ang_vel = torch.tensor(self.omegaBody[np.newaxis, :],device=self.device,dtype=torch.float,requires_grad=False)
        self.dof_pos = torch.tensor([m - n for m,n in zip(self.q,self.default_angles)],device=self.device,dtype=torch.float,requires_grad=False)
        self.projected_gravity = torch.tensor(self.gravity_vector[np.newaxis, :],device=self.device,dtype=torch.float,requires_grad=False)
        self.commands = torch.tensor([forward,side,rotate],device=self.device,dtype=torch.float,requires_grad=False)
        self.dof_vel = torch.tensor([self.dq],device=self.device,dtype=torch.float,requires_grad=False) # 0.5 scale maximum

        self.obs = torch.cat((
            # self.base_lin_vel.squeeze(),
            self.base_ang_vel.squeeze(),
            self.projected_gravity.squeeze(),
            self.commands,
            self.dof_pos,
            self.dof_vel.squeeze(),
            self.actions,
            ),dim=-1)
        
        self.obs = torch.clip(self.obs, -100, 100)

        current_obs = self.obs

        # print("obs shape : ", (self.obs).shape)
        
        self.obs = torch.cat((self.obs,self.obs_storage),dim=-1)
        self.obs_storage[:-self.unit_obs] = self.obs_storage[self.unit_obs:].clone()
        self.obs_storage[-self.unit_obs:] = current_obs

    def init_pose(self):
        while self.init:
            self.pre_step()
            self.get_observations()
            self.motiontime = self.motiontime+1
            if self.motiontime <100:
                self.setJointValues(self.default_angles,kp=5,kd=1)
            else:
                self.setJointValues(self.default_angles,kp=50,kd=5)
            if self.motiontime > 1100:
                self.init = False
            self.post_step()
        print("Starting")
        print("Logging data")
        self.init_log_data = True
        #self.time = 0
        try:
            while True:
                self.pre_step()
                self.finish_rec += self.state.wirelessRemote[2] # L1 button is pressed 
                # print("L1 button : ", self.state.wirelessRemote[2])
                # print("Finish rec : ", self.finish_rec)
                if self.finish_rec >= 2:
                    break
                self.step()
        finally:
            # When the loop is exited (button is pressed), write all collected data to the CSV file
            print("Finished logging data")
            with open(self.csv_filename, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Adjust the column headers based on your joint data
                headers = ['Timestamp'] + [f'Joint {i+1} Angle' for i in range(12)] + [f'Desired Joint {i+1} Angle' for i in range(12)]
                csv_writer.writerow(headers)
                csv_writer.writerows(self.leg_data_buffer)
            print("Finished writing CSV file")

    def pre_step(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)
    
    def step(self):
        '''
        Has to be called after init_pose 
        calls pre_step for getting udp packets
        calls policy with obs, clips and scales actions and adds default pose before sending them to robot
        calls post_step 
        '''
        self.get_observations()
        self.actions = self.policy(self.obs)
        actions = torch.clip(self.actions, -100, 100).to('cpu').detach()
        scaled_actions = actions * 0.25
        final_angles = scaled_actions+self.default_angles_tensor

        if self.init_log_data:
            self.leg_data_buffer.append([self.timestep] + self.q + final_angles.tolist())
        # print("actions:" + ",".join(map(str, actions.numpy().tolist())))
        # print("observations:" + str(time.process_time()) + ",".join(map(str, self.obs.detach().numpy().tolist())))

        self.setJointValues(angles=final_angles,kp=20,kd=0.5)
        # self.setJointValues(self.default_angles,kp=50,kd=5)
        self.post_step()

    def post_step(self):
        '''
        Offers power protection, sends udp packets, maintains timing
        '''
        self.safe.PowerProtect(self.cmd, self.state, 9)
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: 
            print(f"{self.timestep}| frq: {1 / (time.time() - self.time)} Hz")
        self.time = time.time()
        self.timestep = self.timestep + 1

    def getJointVelocity(self):
        velocity = [self.state.motorState[self.d['FL_0']].dq,self.state.motorState[self.d['FL_1']].dq,self.state.motorState[self.d['FL_2']].dq,
                self.state.motorState[self.d['FR_0']].dq,self.state.motorState[self.d['FR_1']].dq,self.state.motorState[self.d['FR_2']].dq,
                self.state.motorState[self.d['RL_0']].dq,self.state.motorState[self.d['RL_1']].dq,self.state.motorState[self.d['RL_2']].dq,
                self.state.motorState[self.d['RR_0']].dq,self.state.motorState[self.d['RR_1']].dq,self.state.motorState[self.d['RR_2']].dq]
        return velocity
    
    def getJointPos(self):
        current_angles = [
        self.state.motorState[self.d['FL_0']].q,self.state.motorState[self.d['FL_1']].q,self.state.motorState[self.d['FL_2']].q,
        self.state.motorState[self.d['FR_0']].q,self.state.motorState[self.d['FR_1']].q,self.state.motorState[self.d['FR_2']].q,
        self.state.motorState[self.d['RL_0']].q,self.state.motorState[self.d['RL_1']].q,self.state.motorState[self.d['RL_2']].q,
        self.state.motorState[self.d['RR_0']].q,self.state.motorState[self.d['RR_1']].q,self.state.motorState[self.d['RR_2']].q]
        return current_angles
    
    def setJointValues(self,angles,kp,kd):
        self.cmd.motorCmd[self.d['FR_0']].q = angles[3]
        self.cmd.motorCmd[self.d['FR_0']].dq = 0
        self.cmd.motorCmd[self.d['FR_0']].Kp = kp
        self.cmd.motorCmd[self.d['FR_0']].Kd = kd
        self.cmd.motorCmd[self.d['FR_0']].tau = 0.0

        self.cmd.motorCmd[self.d['FR_1']].q = angles[4]
        self.cmd.motorCmd[self.d['FR_1']].dq = 0
        self.cmd.motorCmd[self.d['FR_1']].Kp = kp
        self.cmd.motorCmd[self.d['FR_1']].Kd = kd
        self.cmd.motorCmd[self.d['FR_1']].tau = 0.0

        self.cmd.motorCmd[self.d['FR_2']].q = angles[5]
        self.cmd.motorCmd[self.d['FR_2']].dq = 0
        self.cmd.motorCmd[self.d['FR_2']].Kp = kp
        self.cmd.motorCmd[self.d['FR_2']].Kd = kd
        self.cmd.motorCmd[self.d['FR_2']].tau = 0.0

        self.cmd.motorCmd[self.d['FL_0']].q = angles[0]
        self.cmd.motorCmd[self.d['FL_0']].dq = 0
        self.cmd.motorCmd[self.d['FL_0']].Kp = kp
        self.cmd.motorCmd[self.d['FL_0']].Kd = kd
        self.cmd.motorCmd[self.d['FL_0']].tau = 0.0

        self.cmd.motorCmd[self.d['FL_1']].q = angles[1]
        self.cmd.motorCmd[self.d['FL_1']].dq = 0
        self.cmd.motorCmd[self.d['FL_1']].Kp = kp
        self.cmd.motorCmd[self.d['FL_1']].Kd = kd
        self.cmd.motorCmd[self.d['FL_1']].tau = 0.0

        self.cmd.motorCmd[self.d['FL_2']].q = angles[2]
        self.cmd.motorCmd[self.d['FL_2']].dq = 0
        self.cmd.motorCmd[self.d['FL_2']].Kp = kp
        self.cmd.motorCmd[self.d['FL_2']].Kd = kd
        self.cmd.motorCmd[self.d['FL_2']].tau = 0.0

        self.cmd.motorCmd[self.d['RR_0']].q = angles[9]
        self.cmd.motorCmd[self.d['RR_0']].dq = 0
        self.cmd.motorCmd[self.d['RR_0']].Kp = kp
        self.cmd.motorCmd[self.d['RR_0']].Kd = kd
        self.cmd.motorCmd[self.d['RR_0']].tau = 0.0

        self.cmd.motorCmd[self.d['RR_1']].q = angles[10]
        self.cmd.motorCmd[self.d['RR_1']].dq = 0
        self.cmd.motorCmd[self.d['RR_1']].Kp = kp
        self.cmd.motorCmd[self.d['RR_1']].Kd = kd
        self.cmd.motorCmd[self.d['RR_1']].tau = 0.0

        self.cmd.motorCmd[self.d['RR_2']].q = angles[11]
        self.cmd.motorCmd[self.d['RR_2']].dq = 0
        self.cmd.motorCmd[self.d['RR_2']].Kp = kp
        self.cmd.motorCmd[self.d['RR_2']].Kd = kd
        self.cmd.motorCmd[self.d['RR_2']].tau = 0.0

        self.cmd.motorCmd[self.d['RL_0']].q = angles[6]
        self.cmd.motorCmd[self.d['RL_0']].dq = 0
        self.cmd.motorCmd[self.d['RL_0']].Kp = kp
        self.cmd.motorCmd[self.d['RL_0']].Kd = kd
        self.cmd.motorCmd[self.d['RL_0']].tau = 0.0

        self.cmd.motorCmd[self.d['RL_1']].q = angles[7]
        self.cmd.motorCmd[self.d['RL_1']].dq = 0
        self.cmd.motorCmd[self.d['RL_1']].Kp = kp
        self.cmd.motorCmd[self.d['RL_1']].Kd = kd
        self.cmd.motorCmd[self.d['RL_1']].tau = 0.0

        self.cmd.motorCmd[self.d['RL_2']].q = angles[8]
        self.cmd.motorCmd[self.d['RL_2']].dq = 0
        self.cmd.motorCmd[self.d['RL_2']].Kp = kp
        self.cmd.motorCmd[self.d['RL_2']].Kd = kd
        self.cmd.motorCmd[self.d['RL_2']].tau = 0.0

    def getBodyAngularVel(self):
        # self.omegaBody = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (
        #             1 - self.smoothing_ratio) * self.omegaBody

        self.omegaBody = self.smoothing_ratio * np.array(self.state.imu.gyroscope) + (1 - self.smoothing_ratio) * self.omegaBody

        return self.omegaBody
    
    def contactEstimator(self):
        contact_estimate = []
        foot_force = np.array(self.state.footForce)
        
        # Vectorized comparison
        contact_estimate = np.where(foot_force > 10, 0.5, 0)
        
        return contact_estimate
    
    def getBodyAccel(self):
        #Empirical values found with NMPC controller
        x_offset = 0.14641
        y_offset = -0.03673
        alpha = 0.1

        if not self.initialized:
            self.accel = np.array(self.state.imu.accelerometer)
            self.accel[0] -= x_offset
            self.accel[1] -= y_offset
            self.initialized = True
        else:
            offset_input = np.array(self.state.imu.accelerometer)
            offset_input[0] -= x_offset
            offset_input[1] -= y_offset
            self.accel = alpha * offset_input + (1.0 - alpha) * self.accel
        return self.accel
    
    def getQuaternion(self):
        return np.array(self.state.imu.quaternion)

    def quaternionToRotationMatrix(self):
        # compute rBody
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2 * self.quat[2]**2 - 2 * self.quat[3]**2
        R[0, 1] = 2 * self.quat[1] * self.quat[2] - 2 * self.quat[0] * self.quat[3]
        R[0, 2] = 2 * self.quat[1] * self.quat[3] + 2 * self.quat[0] * self.quat[2]
        R[1, 0] = 2 * self.quat[1] * self.quat[2] + 2 * self.quat[0] * self.quat[3]
        R[1, 1] = 1 - 2 * self.quat[1]**2 - 2 * self.quat[3]**2
        R[1, 2] = 2 * self.quat[2] * self.quat[3] - 2 * self.quat[1] * self.quat[0]
        R[2, 0] = 2 * self.quat[1] * self.quat[3] - 2 * self.quat[2] * self.quat[0]
        R[2, 1] = 2 * self.quat[2] * self.quat[3] + 2 * self.quat[1] * self.quat[0]
        R[2, 2] = 1 - 2 * self.quat[1]**2 - 2 * self.quat[2]**2
        return R

    # def cheaterVelocityEstimator(self):
    #     # rBody = self.quaternionToRotationMatrix()
    #     # Rbod = rBody.T
    #     # aWorld = Rbod@self.aBody
    #     # a = aWorld + g
    #     self.base_lin_vel = self.aBody*self.dt
    def get_rotation_matrix_from_rpy(self, rpy):
        """
        Get rotation matrix from the given quaternion.
        Args:
            q (np.array[float[4]]): quaternion [w,x,y,z]
        Returns:
            np.array[float[3,3]]: rotation matrix.
        """
        r, p, y = rpy
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r), -math.sin(r)],
                        [0, math.sin(r), math.cos(r)]
                        ])

        R_y = np.array([[math.cos(p), 0, math.sin(p)],
                        [0, 1, 0],
                        [-math.sin(p), 0, math.cos(p)]
                        ])

        R_z = np.array([[math.cos(y), -math.sin(y), 0],
                        [math.sin(y), math.cos(y), 0],
                        [0, 0, 1]
                        ])

        rot = np.dot(R_z, np.dot(R_y, R_x))
        return rot

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

