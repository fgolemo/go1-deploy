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

from actor_critic import ActorCritic
# from LinearKalmanFilter import LinearKFPositionVelocityEstimator
# from FastLinearKF import FastLinearKFPositionVelocityEstimator

sys.path.append('unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

class Agent():
    def __init__(self,path):
        self.dt = 0.02
        self.num_actions = 12
        self.num_obs = 48 #44*5 #48
        self.unit_obs = 44
        self.num_privl_obs =  self.num_obs #421 # num_obs 
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
#####################################################################
        # dt = 0.002
        self._xhat = np.zeros(18)
        self._xhat[2] = 0.0
        self._ps = np.zeros(12)
        self._vs = np.zeros(12)
        self._A = np.zeros((18, 18))
        self._A[:3, :3] = np.eye(3)
        self._A[:3, 3:6] = self.dt * np.eye(3)
        self._A[3:6, 3:6] = np.eye(3)
        self._A[6:18, 6:18] = np.eye(12)
        self._B = np.zeros((18, 3))
        self._B[3:6, :3] = self.dt * np.eye(3)
        C1 = np.hstack([np.eye(3), np.zeros((3, 3))])
        C2 = np.hstack([np.zeros((3, 3)), np.eye(3)])
        self._C = np.zeros((28, 18))
        for i in range(4):
            self._C[i*3:(i+1)*3, :6] = C1
        self._C[12:15, :6] = C2
        self._C[15:18, :6] = C2
        self._C[18:21, :6] = C2
        self._C[21:24, :6] = C2
        self._C[:12, 6:18] = -1 * np.eye(12)
        self._C[24, 8] = 1
        self._C[25, 11] = 1
        self._C[26, 14] = 1
        self._C[27, 17] = 1
        self._P = 100 * np.eye(18)
        self._Q0 = np.eye(18)
        self._Q0[:3, :3] = (self.dt / 20) * np.eye(3)
        self._Q0[3:6, 3:6] = (self.dt * 9.8 / 20) * np.eye(3)
        self._Q0[6:18, 6:18] = self.dt * np.eye(12)
        self._R0 = np.eye(28)

#####################################################################

        self.default_angles = [0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1,-1.5,-0.1,1,-1.5]
        self.default_angles_tensor = torch.tensor([0.1,0.8,-1.5,-0.1,0.8,-1.5,0.1,1,-1.5,-0.1,1,-1.5],device=self.device,dtype=torch.float,requires_grad=False)
        
        self.actions = torch.zeros(self.num_actions,device=self.device,dtype=torch.float,requires_grad=False)
        self.obs = torch.zeros(self.num_obs,device=self.device,dtype=torch.float,requires_grad=False)
        # self.obs_storage = torch.zeros(self.unit_obs*4,device=self.device,dtype=torch.float)

        actor_critic = ActorCritic(num_actor_obs=self.num_obs,num_critic_obs=self.num_privl_obs,num_actions=12,actor_hidden_dims = [512, 256, 128],critic_hidden_dims = [512, 256, 128],activation = 'elu',init_noise_std = 1.0)
        loaded_dict = torch.load(self.path)
        actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        actor_critic.eval()
        self.policy = actor_critic.act_inference

        self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)

        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

    def get_observations(self):
        # self.euler = np.array(self.state.imu.rpy)
        # self.deuler_history[self.buf_idx % self.smoothing_length, :] = self.euler  - self.euler_prev
        # self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev
        # self.timuprev = time.time()
        # self.buf_idx += 1
        # self.euler_prev = self.euler

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
        self.p, self.v = self.computeLegJacobianPositionVelocityFast(self.q, self.dq)
        self.quat = self.getQuaternion()
        self.omegaBody = self.getBodyAngularVel() #self.state.imu.gyroscope
        self.contact_estimate = self.contactEstimator()
        self.aBody = self.getBodyAccel()
        #vel_estimator = FastLinearKFPositionVelocityEstimator(q, dq, p, v, quat, omegaBody, contact_estimate, body_accel)
        # self.base_lin_vel = self.runKF()
        self.lin_vel = self.aBody*0.001
        self.R = self.get_rotation_matrix_from_rpy(self.state.imu.rpy)
        self.gravity_vector = self.get_gravity_vector()

        # print("gravity : ", self.gravity_vector)
        # print("Foot positions : ", p)
        # print("Foot velocities : ", v)
        # print("Quaternion : ", quat)
        # print("Body ang vel : ", omegaBody)
        # print("Contact estimate : ", contact_estimate)
        # print("Base lin vel : ", self.lin_vel)

        self.pitch = torch.tensor([self.state.imu.rpy[1]],device=self.device,dtype=torch.float,requires_grad=False)
        self.roll = torch.tensor([self.state.imu.rpy[0]],device=self.device,dtype=torch.float,requires_grad=False)
       
        self.dof_pos = torch.tensor([m - n for m,n in zip(self.q,self.default_angles)],device=self.device,dtype=torch.float,requires_grad=False)
        # print(vel[1])
        if self.timestep > 1600:
            self.base_ang_vel = torch.tensor([self.omegaBody],device=self.device,dtype=torch.float,requires_grad=False)
            self.base_lin_vel = torch.tensor([self.lin_vel],device=self.device,dtype=torch.float,requires_grad=False)
            self.projected_gravity = torch.tensor([self.gravity_vector],device=self.device,dtype=torch.float,requires_grad=False)
            self.dof_vel = torch.tensor([self.dq],device=self.device,dtype=torch.float,requires_grad=False)
        else:
            self.base_ang_vel = 0*torch.tensor([self.omegaBody],device=self.device,dtype=torch.float,requires_grad=False)
            self.base_lin_vel = 0*torch.tensor([self.lin_vel],device=self.device,dtype=torch.float,requires_grad=False)
            self.projected_gravity = 0*torch.tensor([self.gravity_vector],device=self.device,dtype=torch.float,requires_grad=False)
            self.dof_vel = 0*torch.tensor([self.dq],device=self.device,dtype=torch.float,requires_grad=False)

        if self.timestep > 2000:
            # self.commands = torch.tensor([0.5,0,0],device=self.device,dtype=torch.float,requires_grad=False)
            self.commands = torch.tensor([forward,side,rotate],device=self.device,dtype=torch.float,requires_grad=False)
        #     print(f"{vel[1]} | {self.base_ang_vel}")
        else:
            self.commands = torch.tensor([0,0,0],device=self.device,dtype=torch.float,requires_grad=False)

        # print("Base ang vel :", self.base_ang_vel)
        # print("Lin vel tensor :", self.base_lin_vel)
        # print("Tensor gravity : ", self.projected_gravity)
        # print("Pitch :", self.pitch)
        # print("Roll :", self.roll)
        # print("Commands :", self.commands)
        # print("Dof pos : ", self.dof_pos)
        # print("Dof vel : ", self.dof_vel)
        # print("Actions : ", self.actions)
        # self.obs = torch.cat((
        #     self.base_ang_vel.squeeze(),
        #     self.pitch,
        #     self.roll,
        #     self.commands,
        #     self.dof_pos,
        #     self.dof_vel.squeeze(),
        #     self.actions,
        #     ),dim=-1)
        self.obs = torch.cat((
            self.base_lin_vel.squeeze(),
            self.base_ang_vel.squeeze(),
            self.projected_gravity.squeeze(),
            self.commands,
            self.dof_pos,
            self.dof_vel.squeeze(),
            self.actions,
            ),dim=-1)
        
        current_obs = self.obs

        #print("obs shape : ", (self.obs).shape)
        
        # self.obs = torch.cat((self.obs,self.obs_storage),dim=-1)

        # self.obs_storage[:-self.unit_obs] = self.obs_storage[self.unit_obs:].clone()
        # self.obs_storage[-self.unit_obs:] = current_obs

    def init_pose(self):
        while self.init:
            self.pre_step()
            self.get_observations()
            self.motiontime = self.motiontime+1
            if self.motiontime <100:
                self.setJointValues(self.default_angles,kp=5,kd=1)
            else:
                self.setJointValues(self.default_angles,kp=50,kd=5)
                # self.setJointValues(self.default_angles,kp=20,kd=0.5)
            if self.motiontime > 1100:
                self.init = False
            self.post_step()
        print("Starting")
        while True:
            self.step()
            # self.get_observations()

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
        self.pre_step()
        self.get_observations()
        self.actions = self.policy(self.obs)
        actions = torch.clip(self.actions, -100, 100).to('cpu').detach()
        scaled_actions = actions * 0.25
        final_angles = scaled_actions+self.default_angles_tensor

        # print("actions:" + ",".join(map(str, actions.numpy().tolist())))
        # print("observations:" + str(time.process_time()) + ",".join(map(str, self.obs.detach().numpy().tolist())))

        self.setJointValues(angles=final_angles,kp=20,kd=0.5)
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

    def computeLegJacobianPositionVelocity(self, q, dq):

        # from const.xacro
        hipLinkLength = 0.08
        thighLinkLength = 0.213
        calfLinkLength = 0.213

        l1 = hipLinkLength
        l2 = thighLinkLength
        l3 = calfLinkLength

        positions = []
        # jacobians = []
        velocities = []

        for leg in range(4):
            sideSign = -1 if leg in [0, 2] else 1

            # Calculate the starting index for the joint angles of the given leg
            start_index = leg * 3
            # Slice the q array to get the joint angles for the specific leg
            leg_q = q[start_index:start_index + 3]
            leg_dq = dq[start_index:start_index + 3]


            s1 = np.sin(leg_q[0])
            s2 = np.sin(leg_q[1])
            s3 = np.sin(leg_q[2])

            c1 = np.cos(leg_q[0])
            c2 = np.cos(leg_q[1])
            c3 = np.cos(leg_q[2])

            c23 = c2 * c3 - s2 * s3
            s23 = s2 * c3 + c2 * s3

            J = np.zeros((3, 3))
            J[0, 0] = 0
            J[1, 0] = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
            J[2, 0] = sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
            J[0, 1] = -l3 * c23 - l2 * c2
            J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
            J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
            J[0, 2] = -l3 * c23
            J[1, 2] = -l3 * s23 * s1
            J[2, 2] = l3 * s23 * c1

            # jacobians.append(J)

            # Position vector calculation for each leg
            p = np.array([
                -calfLinkLength * s23 - thighLinkLength * s2,
                hipLinkLength * sideSign * c1 + calfLinkLength * (s1 * c23) + thighLinkLength * c2 * s1,
                hipLinkLength * sideSign * s1 - calfLinkLength * (c1 * c23) - thighLinkLength * c1 * c2
            ])

            positions.append(p)

            # Compute the leg velocity: v = J * qd
            v = np.dot(J, leg_dq)
            velocities.append(v)

        return positions, velocities

    def computeLegJacobianPositionVelocityFast(self, q, dq):
        # from const.xacro
        hipLinkLength = 0.08
        thighLinkLength = 0.213
        calfLinkLength = 0.213

        # Precompute sin and cos for all angles
        s = np.sin(q)
        c = np.cos(q)

        # Separate sin and cos for each joint
        s1, s2, s3 = s[::3], s[1::3], s[2::3]
        c1, c2, c3 = c[::3], c[1::3], c[2::3]

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        # sideSign for each leg
        sideSign = np.array([-1, 1, -1, 1])

        # Compute positions for all legs
        positions = np.stack([
            -calfLinkLength * s23 - thighLinkLength * s2,
            hipLinkLength * sideSign * c1 + calfLinkLength * s1 * c23 + thighLinkLength * c2 * s1,
            hipLinkLength * sideSign * s1 - calfLinkLength * c1 * c23 - thighLinkLength * c1 * c2
        ], axis=1)

        # Initialize Jacobians and velocities
        J = np.zeros((4, 3, 3))
        velocities = np.zeros((4, 3))

        for leg in range(4):
            # Fill in the Jacobian for each leg
            J[leg, 1, 0] = -sideSign[leg] * hipLinkLength * s1[leg] + thighLinkLength * c2[leg] * c1[leg] + calfLinkLength * c23[leg] * c1[leg]
            J[leg, 2, 0] = sideSign[leg] * hipLinkLength * c1[leg] + thighLinkLength * c2[leg] * s1[leg] + calfLinkLength * c23[leg] * s1[leg]
            J[leg, 0, 1] = -calfLinkLength * c23[leg] - thighLinkLength * c2[leg]
            J[leg, 1, 1] = -thighLinkLength * s2[leg] * s1[leg] - calfLinkLength * s23[leg] * s1[leg]
            J[leg, 2, 1] = thighLinkLength * s2[leg] * c1[leg] + calfLinkLength * s23[leg] * c1[leg]
            J[leg, 0, 2] = -calfLinkLength * c23[leg]
            J[leg, 1, 2] = -calfLinkLength * s23[leg] * s1[leg]
            J[leg, 2, 2] = calfLinkLength * s23[leg] * c1[leg]
             # Compute velocities for each leg
            velocities[leg] = np.dot(J[leg], dq[leg * 3:leg * 3 + 3])
        
        return positions, velocities

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
    
    def getHipLocation(self, leg):
        # assert 0 <= leg < 4

        # Values taken from legged_control/legged_examples/legged_unitree/legged_unitree_description/urdf/go1/const.xacro
        leg_offset_x = 0.1881
        leg_offset_y = 0.04675
        leg_offset_z = 0

        pHip = np.zeros(3)
        
        if leg == 0:
            pHip[0] = leg_offset_x
            pHip[1] = -leg_offset_y
            pHip[2] = leg_offset_z
        elif leg == 1:
            pHip[0] = leg_offset_x
            pHip[1] = leg_offset_y
            pHip[2] = leg_offset_z
        elif leg == 2:
            pHip[0] = -leg_offset_x
            pHip[1] = -leg_offset_y
            pHip[2] = leg_offset_z
        elif leg == 3:
            pHip[0] = -leg_offset_x
            pHip[1] = leg_offset_y
            pHip[2] = leg_offset_z

        return pHip

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


    def runKF(self):
        # orientation : quaternion coming from IMU
        process_noise_pimu = 0.02
        process_noise_vimu = 0.02
        process_noise_pfoot = 0.002
        sensor_noise_pimu_rel_foot = 0.001
        sensor_noise_vimu_rel_foot = 0.1
        sensor_noise_zfoot = 0.001

        Q = np.eye(18)
        Q[:3, :3] *= self._Q0[:3, :3] * process_noise_pimu
        Q[3:6, 3:6] *= self._Q0[3:6, 3:6] * process_noise_vimu
        Q[6:18, 6:18] *= self._Q0[6:18, 6:18] * process_noise_pfoot

        R = np.eye(28)
        R[:12, :12] *= self._R0[:12, :12] * sensor_noise_pimu_rel_foot
        R[12:24, 12:24] *= self._R0[12:24, 12:24] * sensor_noise_vimu_rel_foot
        R[24:28, 24:28] *= self._R0[24:28, 24:28] * sensor_noise_zfoot

        qindex = 0
        rindex1 = 0
        rindex2 = 0
        rindex3 = 0

        g = np.array([0, 0, -9.81])

        # Rotation matrix got from quaternion (w,x,y,z)
        rBody = self.quaternionToRotationMatrix()
        Rbod = rBody.T
        aWorld = Rbod@self.aBody
        a = aWorld + g

        pzs = np.zeros(4)
        trusts = np.zeros(4)
        p0 = self._xhat[:3]
        v0 = self._xhat[3:6]

        for i in range(4):
            i1 = 3 * i
            # quadruped = state_estimator_data.legControllerData.aliengo
            ph = self.getHipLocation(i)
            p_rel = ph + self.p[i]
            dp_rel = self.v[i]
            #p_f = Rbod.dot(p_rel)
            p_f = Rbod @ p_rel
            # dp_f = Rbod.dot(np.cross(self.omegaBody, p_rel) + dp_rel)
            dp_f = Rbod @ (np.cross(self.omegaBody, p_rel) + dp_rel)

            qindex = 6 + i1
            rindex1 = i1
            rindex2 = 12 + i1
            rindex3 = 24 + i

            trust = 1
            phase = min(self.contact_estimate[i], 1)
            trust_window = 0.3

            if phase < trust_window:
                trust = phase / trust_window
            elif phase > (1 - trust_window):
                trust = (1 - phase) / trust_window

            Q[qindex:qindex+3, qindex:qindex+3] *= 1 + (1 - trust) * 100
            R[rindex1:rindex1+3, rindex1:rindex1+3] *= 1
            R[rindex2:rindex2+3, rindex2:rindex2+3] *= 1 + (1 - trust) * 100
            R[rindex3, rindex3] *= 1 + (1 - trust) * 100

            trusts[i] = trust
            self._ps[i1:i1+3] = -p_f
            self._vs[i1:i1+3] = (1.0 - trust) * v0 + trust * (-dp_f)
            pzs[i] = (1.0 - trust) * (p0[2] + p_f[2])

        y = np.concatenate([self._ps, self._vs, pzs])
        # self._xhat = self._A.dot(self._xhat) + self._B.dot(a)
        self._xhat = self._A@self._xhat + self._B@a
        At = self._A.T
        #Pm = self._A.dot(self._P).dot(At) + Q
        Pm = self._A@self._P@At + Q
        Ct = self._C.T
        # yModel = self._C.dot(self._xhat)
        yModel = self._C @ self._xhat
        ey = y - yModel
        # S = self._C.dot(Pm).dot(Ct) + R
        S = self._C @ Pm @ Ct + R

        S_ey = np.linalg.solve(S, ey)
        # self._xhat += Pm.dot(Ct).dot(S_ey)
        self._xhat += Pm @ Ct @ S_ey

        S_C = np.linalg.solve(S, self._C)
        # self._P = (np.eye(18) - Pm.dot(Ct).dot(S_C)).dot(Pm)
        self._P = (np.eye(18) - Pm @  Ct @ S_C) @ Pm

        Pt = self._P.T
        self._P = (self._P + Pt) / 2

        if np.linalg.det(self._P[:2, :2]) > 0.000001:
            self._P[:2, 2:] = 0
            self._P[2:, :2] = 0
            self._P[:2, :2] /= 10

        position = self._xhat[:3]
        vWorld = self._xhat[3:6]
        # vBody = rBody.dot(vWorld)
        vBody = rBody @ vWorld
        return vBody
    
