import numpy as np
import cv2

import zmq
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

from utils.DH_Panda import Panda as Panda_lee
from spatialmath import SE3
from dh_utils.DH_Panda import Panda

from enum import IntEnum
import time

from PIL import Image

import os
import sys
import numpy as np
import argparse
import time
import json
import torch
from torch.utils.data import DataLoader
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R

from robot_simulation import *


class NetProto_Arm(IntEnum):
    LEFT = 0
    RIGHT = 1

class NetProto(IntEnum):
    MOVE_ARM_BY_XYZRPY = 0
    MOVE_ARM_BY_Q = 1
    COMPLETE_MOVE_ARM = 2
    REQ_B2EE = 3
    REP_B2EE = 4
    REQ_Q = 5
    REP_Q = 6
    GRIPPER_OPEN = 7
    GRIPPER_CLOSE = 8
    COMPLETE_GRIPER = 9


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def align_gripper_orientation(transformation, y_target, x_target):
    y_axis = y_target / np.linalg.norm(y_target)
    x_axis = x_target / np.linalg.norm(x_target)
    z_axis = np.cross(x_axis,y_axis)
    x_axis = np.cross(y_axis, z_axis)

    R_new = np.column_stack((x_axis, y_axis, z_axis))

    T_new = np.eye(4)
    T_new[:3,:3] = R_new
    T_new[:3,3] = transformation[:3,3]

    return T_new

if __name__ == '__main__':
    param = np.array(np.load('./calibration/dh.npy'))
    print(param)
    


    context_control = zmq.Context()
    socket_control = context_control.socket(zmq.REQ)
    socket_control.connect("tcp://161.122.114.39:5555") # 눅 ip 

    file_path = '/home/vision/packages/calibration/data/data_train_0528_30/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml'
    file_path = '/home/vision/packages/calibration/data/data_train_0528_30/no_depth/hand_eye_v1_c2b_e2c_single_nominal_mean.yaml'
    file_path = '/home/vision/packages/calibration/data/data_train_0528_30/no_depth/hand_eye_v1_whole_nominal_mean.yaml'
    file_path = '/home/vision/packages/calibration/data/data_train_0528_30/no_depth/hand_eye_v1_whole_single_nominal_mean.yaml'

    calib_info = data_load_calib(file_path)

    tool_rot_45 = True
    calib_info['dh_param'] = np.array(calib_info['dh_param'])
    calib_info['dh_param'][:,[1,2]] /= 1000.
    robot_lee = Panda_lee(calib_info['dh_param'],tool_rot_45, 'm')
    
    ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
    flag = bytes([int(NetProto.REQ_Q)])
    socket_control.send(ARM_flag+flag)
    current_q = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

    base2ee_lee = np.array(robot_lee.fkine(current_q))

    base = np.eye(4)
    H = np.load("/home/vision/packages/calibration/Test_Robot_Manipulation/Client/pt.npy")
    base2cam = np.array(calib_info['c2b'])
    base2cam[:3, 3] /= 1000.0
    base2target = base2cam @ H
    
    base2target[2, 3] += 0.02



    down = np.array([0,0,-1])
    forward = np.array([1,0,0])
    # 엔드이펙터 위 아래 ? 위 방향으로 향하게
    target_z = base2target[:3,2]
    target_y = base2target[:3,1]
    
    base_matrix = np.eye(4)
    base_x = base_matrix[:3,0]
    base_y = base_matrix[:3,1]
    base_z = base_matrix[:3,2]

    dot_down = np.dot(target_z,down)
    dot_foward = np.dot(target_z, forward)
    if dot_down > 0.7: # 그리퍼가 z축아래로 향해 있을 떄
        print("그리퍼 z 아래 향하고")
        # print("griper z down")

        dot = np.dot(target_y,base_x)
        if dot > 0:
            print("그리퍼가 y 바깥을 바라보도록!")
            # print("gipper y outward")
 
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            # base2target_before = base2target_before @ correction
    

    if dot_foward > 0.7: # 그리퍼가 z축 앞으로 향해 있을 떄
        print("그리퍼 z 앞으로 향하고")
        # print("gripper z toward")

        dot = np.dot(target_y,base_z)
        if dot > 0:
            print("그리퍼가 y 위를 바라보도록!")
            # print("gripper y up!")

            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            # base2target_before = base2target_before @ correction

    

    Tep = SE3.CopyFrom(base2target, check=False)
    sol = robot_lee.ik_LM(Tep, q0=current_q)
    pred_matrix = np.array(robot_lee.fkine(sol[0]))
    # pred_matrix[:3, 3] /= 1000.0
    
    print("pred_matrix" ,pred_matrix)
    print("base2target_before" ,base2target)

    distance = np.linalg.norm(base2target[:3,3] - np.array(pred_matrix)[:3,3])
    if distance > 0.05:
        print("Couldn't solve the IK1")
        exit(0)

    if sol[1]:
        q1 = np.array(sol[0], dtype=np.float64)
        
        flag = bytes([int(NetProto.MOVE_ARM_BY_Q)])  # q1 으로 움직임
        socket_control.send(ARM_flag+flag+q1.tobytes())
        # robot_socket.recv()
    
        while True:
            try:
                message = socket_control.recv(zmq.NOBLOCK)
                break
            except zmq.Again:
                print("Wait")

        
        flag = bytes([int(NetProto.GRIPPER_CLOSE)]) # 그리퍼 움직임
        socket_control.send(ARM_flag+flag)

        while True:
            try:
                message = socket_control.recv(zmq.NOBLOCK)
                break
            except zmq.Again:
                print("Wait")

    else:
        print("No Solution")
    


