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

models = [
    "001_handle",           # 1
    "006_mustard_bottle",   # 2
    "trans_cup_1",          # 3
    "trans_cup_2",          # 4
    "trans_cup_3",          # 5
    "water_bottle",         # 6
    "trans_tumbler",        # 7
    "beaker",               # 8
    "sphare_flask",         # 9
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def rtvec_to_matrix(rvec, tvec):
        """
        Convert rotation vector and translation vector to 4x4 matrix
        """
        rvec = np.asarray(rvec)
        tvec = np.asarray(tvec)

        T = np.eye(4)
        R, jac = cv2.Rodrigues(rvec)
        T[:3, :3] = R
        T[:3, 3:] = tvec
        return T

def _draw(img, imgpts):
        origin = imgpts[0].ravel()
        img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)
        img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)
        img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255,0,0), 3)   
        return img


if __name__ == '__main__':

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    nrows = 5
    ncols = 7
    interval=0.029
    axis = np.float32([[0,0,0], [1,0,0],[0,1,0],[0,0,1]]).reshape(-1,3) * interval * 3
    intr = np.eye(3)
    intr[0,0] = intrinsics.fx
    intr[1,1] = intrinsics.fy
    intr[0,2] = intrinsics.ppx
    intr[1,2] = intrinsics.ppy
    print(intr)

    objp = np.zeros((nrows*ncols,3),np.float32)
    objp[:,:2] = np.mgrid[0:ncols,0:nrows].T.reshape(-1,2) * interval
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,300,0.001)
    
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        cv_img = np.asanyarray(color_frame.get_data())[:,:,::-1]

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(nrows,ncols),None)
        # 패턴 보유시 ret = True

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # rotation과 translation 벡터 찾기
            print("self.objp ",objp)
            # print("corners2 ",corners2)
            print("intr ",intr)
        
            _,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp,corners2,intr, np.zeros([4]))

            # 3D 포인트를 이미지 평면에 투영시키자
            imgpts, jac = cv2.projectPoints(axis,rvecs,tvecs,intr, np.zeros([4]))

            # 그리기
            img = cv2.drawChessboardCorners(cv_img.copy(), (ncols, nrows), corners2, ret)
            img = _draw(img, imgpts.astype(np.int64))
            H = rtvec_to_matrix(rvecs, tvecs)

            objp = np.concatenate((objp, np.ones((objp.shape[0], 1))), axis=1) # n x 4
            p3D = (H @ objp.T).T

            # import os
            # base2ee = np.load("/home/vision/packages/robot_operation/base2ee.npy")
            # file_path = "/home/vision/packages/robot_operation/calibration/hand_eye_v1_whole_calib_mean.yaml"
    
            # with open(file_path, 'r') as f:
            #     data = yaml.load(f, yaml.FullLoader)
        
            # base2cam = data['c2b']
            # base2cam = np.array(base2cam)
            # base2cam[:3, 3] /= 1000.0
            # np.set_printoptions(suppress=True)
            # pt = base2cam @ H
            # print(base2ee[:3, 3].astype(np.float16), pt[:3, 3])
            # print("error ",base2ee[:3, 3].astype(np.float16) - pt[:3, 3])
        
            out = {'result': ret, 'c2t': H, 'p2d': np.squeeze(corners, axis=1), 'p3d': p3D[:,:3], 'disp': img}
        else:
            out = {'result': ret, 'disp': cv_img}
        
        # out.write(img)
        cv2.imshow('img', out['disp'])



        key = cv2.waitKey(1)


    param = np.array(np.load('./calibration/dh.npy'))
    print(param)
    
    

    # param = np.array(np.load('./dh_utils/param/dh_param_1024.npy'))
    param = np.array(np.load('./dh_utils/param/dh_param.npy'))
    robot = Panda(param)

    calib_info = data_load_calib(ROOT_DIR)
    cam2base = np.array(calib_info['c2b'])
    cam2base[:3, 3] /= 1000.0

    tool_rot_45 = True
    calib_info['dh_param'] = np.array(calib_info['dh_param'])
    calib_info['dh_param'][:,[1,2]] /= 1000.
    robot_lee = Panda_lee(calib_info['dh_param'],tool_rot_45, 'm')

    

    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                            [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]]).astype(np.float32)
    print("K : ", K)



    context_control = zmq.Context()
    socket_control = context_control.socket(zmq.REQ)
    socket_control.connect("tcp://161.122.114.38:5555")
    
    count = 0

    data_dir = "/home/vision/packages/Scale-Balanced-Grasp"
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

    # 로봇에 맞게 좌표 변경
    half_depth_vector = np.array([0 , 0, 0.109])
    cam2target[:3, 3] = cam2target[:3, 3] + cam2target[:3, :3] @ half_depth_vector
    cam2target_before[:3, 3] = cam2target_before[:3, 3] + cam2target_before[:3, :3] @ half_depth_vector
    
    # 로봇에 맞게 좌표 변경
    half_depth_vector = np.array([0 , 0, -0.10])
    cam2target_before[:3, 3] = cam2target_before[:3, 3] + cam2target_before[:3, :3] @ half_depth_vector
    
    print("==================================================")
    print('cam2target:', cam2target)

    # tvecs1 = pose_matrix[:3,3].reshape(1,3)
    # rvecs1 = R.from_matrix(pose_matrix[:3,:3]).as_rotvec().reshape(1,3)

    # bgr = aligned_color_image[:,:,::-1].copy()
    # cv2.drawFrameAxes(bgr, K, None, rvecs1, tvecs1, 0.05)
    # cv2.imshow('Pose Result', bgr)


    flag = bytes([int(NetProto.REQ_Q)])
    socket_control.send(flag)
    current_q = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

    base2ee = np.array(robot.fkine(current_q))
    base2ee_lee = np.array(robot_lee.fkine(current_q))
    # base2ee_lee[:3, 3] /= 1000.0

    # base2ee[:3, 3] /= 100.0

    print("cam2target ", cam2target)
    print("cam2base ", cam2base)
    print("base2ee ", base2ee)
    print("base2ee_lee ", base2ee_lee)

    np.save("./base2ee",base2ee_lee)
    print("cam2target", cam2target)
    print("cam2target_before", cam2target_before)

    base2target = cam2base @ cam2target 
    # base2target_before = np.linalg.inv(base2ee) @ cam2base @ cam2target_before
    # base2target_before = np.linalg.inv(base2target_before)
    base2target_before = cam2base @ cam2target_before

    from scipy.spatial.transform import Rotation as R
    # 보정 회전: x축 기준 -90도 회전
    # correction = np.eye(4)
    # correction[:3, :3] = R.from_euler('z', 45, degrees=True).as_matrix()

    # base2target = base2target @ correction
    # base2target_before = base2target_before @ correction


    base = np.eye(4)
    

    print("base2target_before ",base2target_before)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    target_frame.transform(base2target_before)
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    target_sphere.paint_uniform_color([0, 0, 1]) # blue
    target_sphere.translate(base2target_before[:3, 3])

    ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    ee_frame.transform(base2ee)
    ee_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    ee_sphere.paint_uniform_color([1, 0, 0]) # red
    # ee_sphere.translate((base2target_before@base2ee)[:3, 3]) # cam2end
    ee_sphere.translate((base2ee)[:3, 3]) # cam2end

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

    base2target_before = base2ee_lee
    base2target_before[:3,3] = [0.72550542 ,0.11320768 ,0.01640262 + 0.05] 
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
    # sol2 = robot_lee.ik_LM(Tep2, q0=q1)
    # pred_matrix2 = np.array(robot_lee.fkine(sol2[0]))
    # pred_matrix2[:3, 3] /= 1000.0

    # distance = np.linalg.norm(base2target[:3,3] - np.array(pred_matrix2)[:3,3])
    # if distance > 0.05:
    #     print("Couldn't solve the IK2")
    # q2 = np.array(sol2[0], dtype=np.float64)

    # ########################################################

    # Tep3 = SE3.CopyFrom(base2target_before, check=False)
    # sol3 = robot_lee.ik_LM(Tep3, q0=q2)
    # pred_matrix3 = robot_lee.fkine(sol3[0])
    # distance = np.linalg.norm(base2target_before[:3,3] - np.array(pred_matrix3)[:3,3])
    # if distance > 0.05:
    #     print("Couldn't solve the IK3")
    # q3 = np.array(sol3[0], dtype=np.float64)

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

    flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q2로 움직임
    socket_control.send(flag+q2.tobytes())
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

    flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q3로 움직임
    socket_control.send(flag+q3.tobytes())
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

    cv2.destroyAllWindows()


    

