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

from datetime import datetime


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




    
    count = 0

    # data_dir = "/home/vision/packages/Scale-Balanced-Grasp"
    # data_dir = "/home/vision/packages/gamma/graspness_implementation"
    data_dir = "/home/vision/packages/FoundationPose/result"

    grasp_info = data_load_grasp(data_dir)
    
    cam2target = grasp_info['transformation']
    cam2target = np.array(cam2target)

    # cam2target_before = grasp_info['transformation_before']
    # cam2target_before = np.array(cam2target_before)
    # from scipy.spatial.transform import Rotation as R
    # # 보정 회전: x축 기준 -90도 회전
    # correction = np.eye(4)
    # correction[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
    # # 보정 적용
    # cam2target_before = cam2target_before @ correction
    # cam2target = cam2target @ correction

    # 로봇에 맞게 좌표 변경 endeffector 끝으로 변경 
    half_depth_vector = np.array([0 , 0, 0.1034])
    cam2target[:3, 3] = cam2target[:3, 3] + cam2target[:3, :3] @ half_depth_vector
    # cam2target_before[:3, 3] = cam2target_before[:3, 3] + cam2target_before[:3, :3] @ half_depth_vector
    


    # 로봇에 맞게 좌표 변경 이전에 몇센치 뒤에 있다가 빠지는 지 
    half_depth_vector = np.array([0 , 0, -0.05])
    # cam2target_before[:3, 3] = cam2target_before[:3, 3] + cam2target_before[:3, :3] @ half_depth_vector
    
    print("==================================================")
    print('cam2target:', cam2target)

    # tvecs1 = pose_matrix[:3,3].reshape(1,3)
    # rvecs1 = R.from_matrix(pose_matrix[:3,:3]).as_rotvec().reshape(1,3)

    # bgr = aligned_color_image[:,:,::-1].copy()
    # cv2.drawFrameAxes(bgr, K, None, rvecs1, tvecs1, 0.05)
    # cv2.imshow('Pose Result', bgr)

    file_path  = "/home/vision/packages/FoundationPose/result/config.json"
    with open(file_path, "rb") as f:
        config_data = json.load(f)


    # context_control_left = zmq.Context()
    # socket_control_left = context_control_left.socket(zmq.REQ)
    # socket_control_left.connect("tcp://161.122.114.39:5555")


    # context_control_right = zmq.Context()
    # socket_control_right = context_control_right.socket(zmq.REQ)
    # socket_control_right.connect("tcp://161.122.114.38:5555")

    context_control = zmq.Context()
    socket_control = context_control.socket(zmq.REQ)
    

    print("+++ check base1 to base2")
    file_path = "/home/vision/packages/calibration/data/data_0424_train_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
    calib_info = data_load_calib(file_path)
    cam2base1 = np.array(calib_info['c2b'])
    
    file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    calib_info = data_load_calib(file_path)
    cam2base2 = np.array(calib_info['c2b'])

    print("base1 to base2")
    print(np.linalg.inv(cam2base1) @ cam2base2)


    if int(config_data["ARM"]) == int(NetProto_Arm.LEFT): 
        ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
        # file_path = "/home/vision/packages/calibration/data/data_0424_train_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
        file_path = "/home/vision/packages/calibration/data/data_0428_left_sum/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
        # socket_control.bind("tcp://161.122.114.33:5555")
        socket_control.connect("tcp://161.122.114.38:5556")
        print("LEFT arm !  ")
    elif int(config_data["ARM"]) == int(NetProto_Arm.RIGHT): 
        ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
        # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
        file_path = "/home/vision/packages/calibration/data/data_0428_right/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
        # socket_control.bind("tcp://161.122.114.34:5556")
        socket_control.connect("tcp://161.122.114.39:5556")
        print("RIGHT arm !")
    

    # ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
    # file_path = "/home/vision/packages/calibration/data/data_0424_train_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
    # # socket_control.bind("tcp://161.122.114.33:5555")
    # socket_control.connect("tcp://161.122.114.38:5556")
    # print("LEFT arm !  ")
    

    # ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
    # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    # # socket_control.bind("tcp://161.122.114.33:5556")
    # socket_control.connect("tcp://161.122.114.39:5556")
    # print("RIGHT arm !")


    

    # ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
    # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    
    calib_info = data_load_calib(file_path)
    cam2base = np.array(calib_info['c2b'])
    cam2base[:3, 3] /= 1000.0

    tool_rot_45 = True
    calib_info['dh_param'] = np.array(calib_info['dh_param'])
    calib_info['dh_param'][:,[1,2]] /= 1000.
    robot_lee = Panda_lee(calib_info['dh_param'],tool_rot_45, 'm')


    # base2ee_lee[:3, 3] /= 1000.0

    # base2ee[:3, 3] /= 100.0

    print("cam2target ", cam2target)
    print("cam2base ", cam2base)

    print("cam2target", cam2target)
    # print("cam2target_before", cam2target_before)

    base2target = cam2base @ cam2target 
    # base2target_before = np.linalg.inv(base2ee) @ cam2base @ cam2target_before
    # base2target_before = np.linalg.inv(base2target_before)
    # base2target_before = cam2base @ cam2target_before


    y_up = np.array([0,-1,0])
    x_out = np.array([1,0,0])

    # base2target = align_gripper_orientation(base2target, y_up, x_out)
    # base2target_before = align_gripper_orientation(base2target_before, y_up, x_out)

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
        # print("그리퍼 z 아래 향하고")
        print("griper z down")

        dot = np.dot(target_y,base_x)
        if dot > 0:
            # print("그리퍼가 y 바깥을 바라보도록!")
            print("gipper y outward")
 
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            # base2target_before = base2target_before @ correction
    

    if dot_foward > 0.7: # 그리퍼가 z축 앞으로 향해 있을 떄
        # print("그리퍼 z 앞으로 향하고")
        print("gripper z toward")

        dot = np.dot(target_y,base_z)
        if dot > 0:
            # print("그리퍼가 y 위를 바라보도록!")
            print("gripper y up!")

            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            # base2target_before = base2target_before @ correction


        
    # base_matrix = np.eye(4)
    # base_z = base_matrix[:3,2]
    target_y = base2target[:3,1]

    # dot = np.dot(base_z, target_y)
    # if dot > 0:
    #     print("그리퍼 위쪽으로 향하게!!!!!!!!!!!!!!!!")
        
    #     correction = np.eye(4)
    #     correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
    #     base2target = base2target @ correction
    #     base2target_before = base2target_before @ correction
        
    #     # base2target = base2target @ flip_y
    #     # base2target_before = base2target_before @ flip_y
    # else:
    #     # ee 바깥쪽으로 향하게
    #     base_x = base_matrix[:3,1]
    #     target_x = base2target[:3,1]
    #     dot = np.dot(base_x, target_y)

    #     if dot > 0:   
    #         print("그리퍼 바깥쪽으로 향하게!!!!!!!!!!!!!!!!")

    #         correction = np.eye(4)
    #         correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
    #         base2target = base2target @ correction
    #         base2target_before = base2target_before @ correction

    # tool 45 false 시 사용
    # correction = np.eye(4)
    # correction[:3, :3] = R.from_euler('z', -45, degrees=True).as_matrix()
    # base2target = base2target @ correction
    # base2target_before = base2target_before @ correction
    

        


    # base2target = np.eye(4)
    # base2target = np.array([[0.0000001, 1.0000001 , 0.000001 , 0.000001],[1.0000001 , 0.000001 , 0.000001 ,0.000001],[0.000001 , 0.000001, -1.000001 ,  0.000001],[0.0, 0.0 , 0.0 ,1.0]])
    # base2target_chess = np.load("/home/vision/packages/calibration/Test_Robot_Manipulation/Client/pt.npy")
    # print("base2target_chess ",base2target_chess)
    # base2target[:3,3] = base2target_chess[:3,3]

    # correction = np.eye(4)
    # correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
    # base2target = base2target @ correction

    # base2target[2, 3] += 0.01


    print("base2target ",base2target)




    # print("base2target_before ",base2target_before)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    target_frame.transform(base2target)
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    target_sphere.paint_uniform_color([0, 0, 1]) # blue
    target_sphere.translate(base2target[:3, 3])

    # ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    # ee_frame.transform(base2ee_lee)
    # ee_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # ee_sphere.paint_uniform_color([1, 0, 0]) # red
    # # ee_sphere.translate((base2target_before@base2ee)[:3, 3]) # cam2end
    # ee_sphere.translate((base2ee_lee)[:3, 3]) # cam2end

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    cam_frame.transform(cam2base)
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    cam_sphere.paint_uniform_color([0, 1, 0]) # green
    cam_sphere.translate(cam2base[:3, 3])


    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    base_sphere.paint_uniform_color([0, 0, 0]) # black


    # o3d.visualization.draw_geometries([base_frame,ee_frame, cam_frame, target_frame,ee_sphere,base_sphere,cam_sphere, target_sphere])
    # o3d.visualization.draw_geometries([base_frame, cam_frame, target_frame,base_sphere,cam_sphere, target_sphere])

    from spatialmath import SE3
    # print(base2target.shape)
    # e2c_2 = SE3(np.array(base2target))
    # rpy = e2c_2.rpy(unit='rad',order='zyx')
    # print("rpy ",rpy)



    trans = base2target.flatten()

    # file_path  = "./subin.json"
    # with open(file_path, "rb") as f:
    #     data = json.load(f)

    # print(data["end_effector_position"])

    # positions = np.array(data["end_effector_position"])
    # rotations = np.array(data["end_effector_rotation"])

    # transformation_metrices = []

    # for i in range(len(positions)):
    #     euler = rotations[i]
    #     pos = positions[i]

    #     rot_matrix = R.from_euler("xyz", euler).as_matrix()

    #     T = np.eye(4)
    #     T[:3,:3] = rot_matrix
    #     T[:3,3] = pos

    #     transformation_metrices.append(T)

    # # print(transformation_metrices[0])
    # trans = transformation_metrices[1].flatten()
    
    # print("전송하는 Trans 값 : ", trans)
    print("Byte : ", trans.tobytes())
    socket_control.send(trans.tobytes())
    # robot_socket.recv()

    TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"./logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    filename = f'./logs/{TIME}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    frame_size = (640, 480)

    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)


    while True:
        # try:
            # message = socket_control.recv(zmq.NOBLOCK)
            # break
        # except zmq.Again:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())[:,:,::-1]
        out.write(img)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)

        if key == ord('y'):
            break

    out.release()
    cv2.destroyAllWindows()
