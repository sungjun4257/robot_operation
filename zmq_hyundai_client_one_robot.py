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
    data_dir = "/home/vision/packages/gamma/graspness_implementation"
    grasp_info = data_load_grasp(data_dir)
    
    cam2target = grasp_info['transformation']
    cam2target = np.array(cam2target)

    cam2target_before = grasp_info['transformation_before']
    cam2target_before = np.array(cam2target_before)
    # from scipy.spatial.transform import Rotation as R
    # # 보정 회전: x축 기준 -90도 회전
    # correction = np.eye(4)
    # correction[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
    # # 보정 적용
    # cam2target_before = cam2target_before @ correction
    # cam2target = cam2target @ correction

    # 로봇에 맞게 좌표 변경 endeffector 끝으로 변경 
    half_depth_vector = np.array([0 , 0, 0.109])
    cam2target[:3, 3] = cam2target[:3, 3] + cam2target[:3, :3] @ half_depth_vector
    cam2target_before[:3, 3] = cam2target_before[:3, 3] + cam2target_before[:3, :3] @ half_depth_vector
    


    # 로봇에 맞게 좌표 변경 이전에 몇센치 뒤에 있다가 빠지는 지 
    half_depth_vector = np.array([0 , 0, -0.05])
    cam2target_before[:3, 3] = cam2target_before[:3, 3] + cam2target_before[:3, :3] @ half_depth_vector
    
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


    # ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
    # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    context_control = zmq.Context()
    socket_control = context_control.socket(zmq.REQ)

    # file_path = "/home/vision/packages/calibration/data/data_0424_train_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
    file_path = "/home/vision/packages/calibration/data/data_0428_left/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    # # socket_control.bind("tcp://161.122.114.33:5555")
    socket_control.connect("tcp://161.122.114.38:5556")
    print("LEFT arm !  ")

    # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_whole_nominal_mean.yaml"
    # socket_control.bind("tcp://161.122.114.34:5556")
    # socket_control.connect("tcp://161.122.114.39:5556")
    # print("RIGHT arm !")
    
    calib_info = data_load_calib(file_path)
    cam2base = np.array(calib_info['c2b'])
    cam2base[:3, 3] /= 1000.0

    tool_rot_45 = True
    calib_info['dh_param'] = np.array(calib_info['dh_param'])
    calib_info['dh_param'][:,[1,2]] /= 1000.
    robot_lee = Panda_lee(calib_info['dh_param'],tool_rot_45, 'm')

    flag = bytes([int(NetProto.REQ_Q)])
    socket_control.send(flag)
    current_q = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

    base2ee_lee = np.array(robot_lee.fkine(current_q))
    # base2ee_lee[:3, 3] /= 1000.0

    # base2ee[:3, 3] /= 100.0

    print("cam2target ", cam2target)
    print("cam2base ", cam2base)
    print("base2ee_lee ", base2ee_lee)

    np.save("./base2ee",base2ee_lee)
    print("cam2target", cam2target)
    print("cam2target_before", cam2target_before)

    base2target = cam2base @ cam2target 
    # base2target_before = np.linalg.inv(base2ee) @ cam2base @ cam2target_before
    # base2target_before = np.linalg.inv(base2target_before)
    base2target_before = cam2base @ cam2target_before



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
        print("그리퍼 z 아래 향하고")
        dot = np.dot(target_y,base_x)
        if dot > 0:
            print("그리퍼가 y 바깥을 바라보도록!")
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            base2target_before = base2target_before @ correction
    

    if dot_foward > 0.7: # 그리퍼가 z축 앞으로 향해 있을 떄
        print("그리퍼 z 앞으로 향하고")
        dot = np.dot(target_y,base_z)
        if dot > 0:
            print("그리퍼가 y 위를 바라보도록!")
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            base2target_before = base2target_before @ correction


        
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
    

        


    base = np.eye(4)

    base2target_chess = np.load("/home/vision/packages/calibration/Test_Robot_Manipulation/Client/pt.npy")
    base2target_before = base2ee_lee
    base2target_before[:3,3] = base2target_chess[:3,3]
    # base2target_before[2, 3] += 0.01
    

    

    print("base2target_before ",base2target_before)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    target_frame.transform(base2target)
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    target_sphere.paint_uniform_color([0, 0, 1]) # blue
    target_sphere.translate(base2target[:3, 3])

    ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    ee_frame.transform(base2ee_lee)
    ee_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    ee_sphere.paint_uniform_color([1, 0, 0]) # red
    # ee_sphere.translate((base2target_before@base2ee)[:3, 3]) # cam2end
    ee_sphere.translate((base2ee_lee)[:3, 3]) # cam2end

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    cam_frame.transform(cam2base)
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    cam_sphere.paint_uniform_color([0, 1, 0]) # green
    cam_sphere.translate(cam2base[:3, 3])


    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    base_sphere.paint_uniform_color([0, 0, 0]) # black


    o3d.visualization.draw_geometries([base_frame,ee_frame, cam_frame, target_frame,ee_sphere,base_sphere,cam_sphere, target_sphere])
    # base2target[0, 3] -= 0.01
    # base2target[2, 3] += 0.02

    # target2base = np.linalg.inv(base2target)
    # target2base[2, 3] += 0.1
    # base2target_pre = np.linalg.inv(target2base)

    # base2target_after = base2target.copy()
    # base2target_after[2, 3] += 0.07

    # Calibration 확인용
    # base2target_before = base2ee_lee
    # # base2target_before[:3,3] = [0.72550542 ,0.11320768 ,0.01640262 + 0.05] # 캘리브레이션 할떄 이 좌표로 잘 가는지 확인용
    # # base2target_before[:3,3] = [0.52682149, 0.13366197 , 0.2678644+ 0.01]
    # base2target_before[:3,3] = [0.11378226 , -0.12694568  , 0.38849961]



    Tep = SE3.CopyFrom(base2target_before, check=False)
    sol = robot_lee.ik_LM(Tep, q0=current_q)
    pred_matrix = np.array(robot_lee.fkine(sol[0]))
    # pred_matrix[:3, 3] /= 1000.0
    
    print("pred_matrix" ,pred_matrix)

    distance = np.linalg.norm(base2target_before[:3,3] - np.array(pred_matrix)[:3,3])
    if distance > 0.05:
        print("Couldn't solve the IK1")
        exit(0)
    q1 = np.array(sol[0], dtype=np.float64)

    ####################################################3

    Tep2 = SE3.CopyFrom(base2target, check=False)
    sol2 = robot_lee.ik_LM(Tep2, q0=q1)
    pred_matrix2 = np.array(robot_lee.fkine(sol2[0]))
    # pred_matrix2[:3, 3] /= 1000.0

    # distance = np.linalg.norm(base2target[:3,3] - np.array(pred_matrix2)[:3,3])
    # if distance > 0.05:
    #     print("Couldn't solve the IK2")
    #     exit(0)
    # q2 = np.array(sol2[0], dtype=np.float64)

    # #######################################################

    # Tep3 = SE3.CopyFrom(base2target_before, check=False)
    # sol3 = robot_lee.ik_LM(Tep3, q0=q2)
    # pred_matrix3 = robot_lee.fkine(sol3[0])
    # distance = np.linalg.norm(base2target_before[:3,3] - np.array(pred_matrix3)[:3,3])
    # if distance > 0.05:
    #     print("Couldn't solve the IK3")
    #     exit(0)
    # q3 = np.array(sol3[0], dtype=np.float64)

    # #########################################################
    # Tep4 = SE3.CopyFrom(base2ee_lee, check=False)
    # sol4 = robot_lee.ik_LM(Tep4, q0=q3)
    # pred_matrix4 = robot_lee.fkine(sol4[0])
    # distance = np.linalg.norm(base2ee_lee[:3,3] - np.array(pred_matrix4)[:3,3])
    # if distance > 0.05:
    #     print("Couldn't solve the IK4")
    #     exit(0)
    # q4 = np.array(sol4[0], dtype=np.float64)

    # filename = f'./tmp/{models[obj_type-1]}.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = 30.0
    # frame_size = (640, 480)

    # out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    flag = bytes([int(NetProto.MOVE_ARM_BY_Q)])  # q1 으로 움직임
    socket_control.send(flag+q1.tobytes())
    # robot_socket.recv()

    while True:
        try:
            message = socket_control.recv(zmq.NOBLOCK)
            break
        except zmq.Again:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())[:,:,::-1]
            # out.write(img)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)

    # flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q2로 움직임
    # socket_control.send(ARM_flag+flag+q2.tobytes())
    # # robot_socket.recv()

    # while True:
    #     try:
    #         message = socket_control.recv(zmq.NOBLOCK)
    #         break
    #     except zmq.Again:
    #         frames = pipeline.wait_for_frames()
    #         color_frame = frames.get_color_frame()

    #         if not color_frame:
    #             continue

    #         img = np.asanyarray(color_frame.get_data())[:,:,::-1]
    #         # out.write(img)
    #         cv2.imshow('img', img)
    #         key = cv2.waitKey(1)

    flag = bytes([int(NetProto.GRIPPER_CLOSE)]) # 그리퍼 움직임
    socket_control.send(flag)
    # robot_socket.recv()

    while True:
        try:
            message = socket_control.recv(zmq.NOBLOCK)
            break
        except zmq.Again:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())[:,:,::-1]
            # out.write(img)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)

    # flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q3로 움직임
    # socket_control.send(ARM_flag+flag+q3.tobytes())
    # # robot_socket.recv()

    # while True:
    #     try:
    #         message = socket_control.recv(zmq.NOBLOCK)
    #         break
    #     except zmq.Again:
    #         frames = pipeline.wait_for_frames()
    #         color_frame = frames.get_color_frame()

    #         if not color_frame:
    #             continue

    #         img = np.asanyarray(color_frame.get_data())[:,:,::-1]
    #         # out.write(img)
    #         cv2.imshow('img', img)
    #         key = cv2.waitKey(1)

    # flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q3로 움직임
    # socket_control.send(ARM_flag+flag+current_q.tobytes())
    # # robot_socket.recv()

    # while True:
    #     try:
    #         message = socket_control.recv(zmq.NOBLOCK)
    #         break
    #     except zmq.Again:
    #         frames = pipeline.wait_for_frames()
    #         color_frame = frames.get_color_frame()

    #         if not color_frame:
    #             continue

    #         img = np.asanyarray(color_frame.get_data())[:,:,::-1]
    #         # out.write(img)
    #         cv2.imshow('img', img)
    #         key = cv2.waitKey(1)

    # flag = bytes([int(NetProto.GRIPPER_OPEN)]) # 그리퍼 움직임
    # socket_control.send(ARM_flag+flag)
    # # robot_socket.recv()

    # while True:
    #     try:
    #         message = socket_control.recv(zmq.NOBLOCK)
    #         break
    #     except zmq.Again:
    #         frames = pipeline.wait_for_frames()
    #         color_frame = frames.get_color_frame()

    #         if not color_frame:
    #             continue

    #         img = np.asanyarray(color_frame.get_data())[:,:,::-1]
    #         # out.write(img)
    #         cv2.imshow('img', img)
    #         key = cv2.waitKey(1)


    # cv2.destroyAllWindows()


    

