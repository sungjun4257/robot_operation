import numpy as np
import cv2

import zmq
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

from utils.DH_Panda import Panda
from spatialmath import SE3

from enum import IntEnum

import os
import numpy as np
import argparse
import time
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from robot_simulation import *



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
    

    

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]]).astype(np.float32)
    print("K : ", K)



    context_control = zmq.Context()
    socket_control = context_control.socket(zmq.REQ)

    socket_control.connect("tcp://161.122.114.38:5556") # 로봇 PC와의 IP, PORT 맞춰줘야함

    file_path = "/home/vision/packages/calibration/data/data_0428_left/no_depth/hand_eye_v1_c2b_nominal_mean.yaml" # DH 파라미터 불러오기 위해
    with open(file_path, 'r') as f:
        calib_info = yaml.load(f, yaml.FullLoader)

    tool_rot_45 = True
    robot = Panda(calib_info['dh_param'],tool_rot_45, 'm') # 현재 사용하는 로봇 객체 정의

    flag = bytes([int(NetProto.REQ_Q)])
    socket_control.send(flag)
    current_q = np.frombuffer(socket_control.recv()[1:], dtype=np.float64) # 현재 q값 받아옴

    base2target = None
    Tep = SE3.CopyFrom(base2target, check=False) # 내가 가고자 하는 base 기준 target의 4x4 transformation matrix (단위는 m)  
    sol = robot.ik_LM(Tep, q0=current_q) # IK 푼다 
    pred_matrix = np.array(robot.fkine(sol[0])) # IK를 푼 관절값으로 다시 FK 함 (오류 확인용)
    
    distance = np.linalg.norm(base2target[:3,3] - np.array(pred_matrix)[:3,3]) # IK 푼 값과 FK 다시 적용한 값 비교
    if distance > 0.05:
        print("Couldn't solve the IK1")
        exit(0)
    q1 = np.array(sol[0], dtype=np.float64)


    flag = bytes([int(NetProto.MOVE_ARM_BY_Q)])  # q1 으로 움직임
    socket_control.send(flag+q1.tobytes()) # flag와 q1값 넣어서 보냄 

    # 로봇이 동작하는 동안 이미지 확인용 및 로봇이 동작을 다 했으면 정지
    while True:
        try:
            message = socket_control.recv(zmq.NOBLOCK) #로봇이 동작을 다 했을 때 이 메세지 받아서 종료
            break
        except zmq.Again:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())[:,:,::-1]
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
