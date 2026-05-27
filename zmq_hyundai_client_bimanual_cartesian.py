import numpy as np
import cv2

import zmq
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import copy
from utils.DH_Panda import Panda as Panda_lee
from spatialmath import SE3
from dh_utils.DH_Panda import Panda

from enum import IntEnum

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
from scipy.spatial.transform import Rotation as R, Slerp
from datetime import datetime

from robot_simulation import *

np.set_printoptions(suppress=True)   # 지수 표기 억제

class NetProto_Arm(IntEnum):
    LEFT = 0
    RIGHT = 1

class NetProto(IntEnum):
    MOVE_ARM_BY_XYZRPY = 0
    MOVE_ARM_BY_Q = 1
    COMPLETE_MOVE_ARM = 2
    REQ_B2EE = 3
    REQ_XYZROLLPITCHYAW = 4
    REQ_Q = 5
    REP_Q = 6
    GRIPPER_OPEN = 7
    GRIPPER_CLOSE = 8
    COMPLETE_GRIPER = 9

TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

filename = f'/home/vision/packages/FoundationPose/result/video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
frame_size = (640, 480)

out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

def solve_ik_and_validate(robot, target_matrix, q_init, name=""):
    Tep = SE3.CopyFrom(target_matrix, check=False)
    sol = robot.ik_LM(Tep, q0=q_init)
    pred_matrix = np.array(robot.fkine(sol[0]))
    
    distance = np.linalg.norm(target_matrix[:3, 3] - pred_matrix[:3, 3])
    if distance > 0.05:
        print(f"Couldn't solve the IK: {name}")
        exit(0)
    
    return np.array(sol[0], dtype=np.float64)


def send_and_wait_for_motion(q, socket_control, flag_code, ARM_flag, pipeline, motion_time):
    flag = bytes([int(flag_code)])
    time = np.array([motion_time])
    print("time" , time)
    socket_control.send(ARM_flag + flag + q.tobytes() + time.tobytes())

    # start_time = time.time()
    timeout = 30

    while True:
        # if time.time() - start_time > timeout:
        #     print(f"⚠️ Timeout ({timeout}s) reached — saving and exiting.")
        #     out.release()
        #     break

        try:
            message = socket_control.recv(zmq.NOBLOCK)
            break
        except zmq.Again:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())[:, :, ::-1]
            out.write(img)

            cv2.imshow('img', img)
            cv2.waitKey(1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # q 키 누르면 종료
                print("🛑 'q' pressed — stopping and saving video.")
                out.release()
                break


def wait_for_gripper_action(socket_control, flag_code, ARM_flag, pipeline):
    flag = bytes([int(flag_code)])
    socket_control.send(ARM_flag + flag)

    while True:
        try:
            message = socket_control.recv(zmq.NOBLOCK)
            break
        except zmq.Again:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())[:, :, ::-1]
            out.write(img)

            cv2.imshow('img', img)
            cv2.waitKey(1)


def is_valid_rotation_matrix(R_mat, atol=1e-6):
    """
    회전 행렬 유효성 검사
    R_mat: 3x3 numpy array
    return: bool
    """
    if R_mat.shape != (3, 3):
        return False

    # 직교성 검사
    should_be_identity = np.dot(R_mat.T, R_mat)
    I = np.eye(3)
    is_orthogonal = np.allclose(should_be_identity, I, atol=atol)

    # 행렬식 검사
    has_det_one = np.isclose(np.linalg.det(R_mat), 1.0, atol=atol)

    return is_orthogonal and has_det_one


def orthonormalize_rotation(R_mat):
    """
    SVD를 이용해 가장 가까운 올바른 회전 행렬로 보정
    """
    U, _, Vt = np.linalg.svd(R_mat)
    R_ortho = U @ Vt

    # 반사 행렬 방지(det < 0인 경우 처리)
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt

    return R_ortho


def interpolate_xyzrpy(current_xyzrpy, target_xyzrpy, alpha=0.5):
    """
    현재 포즈(current_xyzrpy)와 목표 포즈(target_xyzrpy) 사이를 보간하여
    via_matrix_xyzrpy를 생성
    alpha: 보간 비율 (0.0 ~ 1.0)
    return: 보간된 XYZRPY
    """
    # --- 1) XYZ 위치 선형 보간 ---
    via_xyz = (1 - alpha) * current_xyzrpy[:3] + alpha * target_xyzrpy[:3]

    # --- 2) 회전 SLERP 보간 ---
    # 현재 & 목표 회전 행렬
    R_current = R.from_euler('xyz', current_xyzrpy[3:], degrees=False)
    R_target = R.from_euler('xyz', target_xyzrpy[3:], degrees=False)

    key_rots = R.from_matrix([R_current.as_matrix(), R_target.as_matrix()])
    slerp = Slerp([0, 1], key_rots)
    R_via = slerp([alpha]).as_matrix()[0]

    # --- 3) 회전 행렬 유효성 검사 및 보정 ---
    if not is_valid_rotation_matrix(R_via):
        print("⚠️ 보간된 회전 행렬이 비정상입니다. 보정 수행!")
        R_via = orthonormalize_rotation(R_via)
    else:
        print("✅ 보간된 회전 행렬이 정상입니다.")

    # --- 4) 보간된 회전 행렬 → RPY 변환 ---
    via_rpy = R.from_matrix(R_via).as_euler('xyz', degrees=False)

    # --- 5) XYZ + RPY 합치기 ---
    return np.concatenate([via_xyz, via_rpy])

def offset_y(xyzrpy, arm_flag):
    if int(arm_flag) == int(NetProto_Arm.LEFT): # left
        print("LEFT")
        xyzrpy[1] = xyzrpy[1] + 0.3
        # 0.39706217 -0.08047105  0.40784926
    else:
        print("RIGHT")
        xyzrpy[1] = xyzrpy[1] - 0.3

    return xyzrpy

def mat_to_xyzrpw(matrix):
    # 위치 추출
    x, y, z = matrix[:3, 3]
    # 회전 추출 (roll, pitch, yaw)
    r = R.from_matrix(matrix[:3, :3])
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)  # 라디안 단위
    # XYZ + RPY 합치기
    return np.array([x, y, z, roll, pitch, yaw])

def xyzrpw_to_mat(xyzrpw, degrees=False):
    """
    XYZRPW를 4x4 Homogeneous Transformation Matrix로 변환
    xyzrpw: [x, y, z, roll, pitch, yaw]
    degrees: True면 roll, pitch, yaw 단위를 degree로 간주
    """
    # 위치 추출
    x, y, z = xyzrpw[:3]
    roll, pitch, yaw = xyzrpw[3:]

    # 회전 행렬 생성
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    rot_mat = r.as_matrix()

    # 4x4 변환 행렬 생성
    mat = np.eye(4)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = [x, y, z]

    return mat

# def compute_motion_time(current_xyzrpy, target_xyzrpy,
#                         max_linear_speed=0.05, max_angular_speed=0.5):
def compute_motion_time(current_xyzrpy, target_xyzrpy,
                        max_linear_speed=0.1, max_angular_speed=0.5):
    """
    XYZRPY 두 pose를 이용해 motion time 계산
    :param current_xyzrpy: 현재 pose [x, y, z, roll, pitch, yaw] (m, rad)
    :param target_xyzrpy: 목표 pose [x, y, z, roll, pitch, yaw] (m, rad)
    :param max_linear_speed: 최대 선속도 [m/s]
    :param max_angular_speed: 최대 각속도 [rad/s]
    :return: motion_time [s]
    """
    # --- 1) 위치 차이 ---
    pos_diff = np.linalg.norm(target_xyzrpy[:3] - current_xyzrpy[:3])

    # --- 2) 회전 차이 ---
    R_current = R.from_euler('xyz', current_xyzrpy[3:], degrees=False)
    R_target = R.from_euler('xyz', target_xyzrpy[3:], degrees=False)
    R_diff = R_target * R_current.inv()
    rot_diff = R_diff.magnitude()

    # --- 3) 선속도, 각속도 기반 motion time ---
    time_pos = pos_diff / max_linear_speed
    time_rot = rot_diff / max_angular_speed
    print("time_pos ",time_pos)
    print("time_rot ",time_rot)
    motion_time = max(time_pos, time_rot)

    return motion_time

if __name__ == '__main__':

    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--text_prompt', type=str , help='Model checkpoint path', default=None, required=False)
    # cfgs = parser.parse_args()
    # object_name = cfgs.text_prompt

    

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
    socket_control.connect("tcp://161.122.114.39:5555")
    
    count = 0

    # data_dir = "/home/vision/packages/Scale-Balanced-Grasp"
    data_dir = "/home/vision/packages/FoundationPose/result"
    grasp_info = data_load_grasp(data_dir)
    
    cam2target = grasp_info['transformation']
    cam2target = np.array(cam2target)

    cam2target_before = grasp_info['transformation']
    cam2target_before = np.array(cam2target)
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

    half_depth_vector = np.array([0 , 0, -0.08])
    cam2target_before = copy.deepcopy(cam2target)
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

    object_name = config_data["object_name"]
    print(object_name)

    arm_flag = grasp_info['arm']
    print("arm_flag ",arm_flag)

    if int(arm_flag) == int(NetProto_Arm.LEFT): 
        ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
        # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
        file_path = "/home/vision/packages/calibration/data/data_1030_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
    elif int(arm_flag) == int(NetProto_Arm.RIGHT): 
        ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
        # file_path = "/home/vision/packages/calibration/data/data_train_0528_30/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
        file_path = "/home/vision/packages/calibration/data/data_1030_right/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
        
    # ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
    # file_path = "/home/vision/packages/calibration/data/data_train_0528_30/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
    
    # ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
    # file_path = "/home/vision/packages/calibration/data/data_0416_new/no_depth/hand_eye_v1_c2b_nominal_mean.yaml"
    
    calib_info = data_load_calib(file_path)
    cam2base = np.array(calib_info['c2b'])
    cam2base[:3, 3] /= 1000.0

    # tool_rot_45 = True
    # calib_info['dh_param'] = np.array(calib_info['dh_param'])
    # calib_info['dh_param'][:,[1,2]] /= 1000.
    # robot_lee = Panda_lee(calib_info['dh_param'],tool_rot_45, 'm')

    flag = bytes([int(NetProto.REQ_XYZROLLPITCHYAW)])
    socket_control.send(ARM_flag+flag)
    current_xyzrpw = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)
    print("current_xyzrpw ",current_xyzrpw)
    # exit(0)
    # base2ee_lee[:3, 3] /= 1000.0

    # base2ee[:3, 3] /= 100.0

    print("cam2target ", cam2target)
    print("cam2base ", cam2base)

    print("cam2target", cam2target)

    base2target = cam2base @ cam2target 
    base2target_before = cam2base @ cam2target_before
    # base2target_before = np.linalg.inv(base2ee) @ cam2base @ cam2target_before
    # base2target_before = np.linalg.inv(base2target_before)


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
    
    print("dot_down ", dot_down)
    if (dot_down > 0.7 or (dot_down > -0.1 and dot_down < 0.1)): # 그리퍼가 z축아래로 향해 있을 떄
        print("그리퍼 z 아래 향하고")
        dot = np.dot(target_y,base_x)
        print(dot)
        if dot > 0:
            print("그리퍼가 y 바깥을 바라보도록!")
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            base2target_before = base2target_before @ correction
    else:
        print("그리퍼 아래로 잘 향해 있음. dot_down : ",dot_down)
    
    target_z = base2target[:3,2]
    target_y = base2target[:3,1]
    dot_foward = np.dot(target_z, forward)
    print("dot forward ",dot_foward)
    if dot_foward > 0.7: # 그리퍼가 z축 앞으로 향해 있을 떄
        print("그리퍼 z 앞으로 향하고")
        dot = np.dot(target_y,base_z)
        if dot > 0:
            print("그리퍼가 y 위를 바라보도록!")
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
            base2target_before = base2target_before @ correction
    else:
        print("그리퍼 앞으로 잘 향해 있음. dot_down : ",dot_foward)

    correction = np.eye(4)
    correction[:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix() 
    base2target_before = base2target_before @ correction
    base2target = base2target @ correction

    base2ee = xyzrpw_to_mat(current_xyzrpw)
    base2target_xyzrpy = mat_to_xyzrpw(base2target)
    base2target_before_xyzrpy = mat_to_xyzrpw(base2target_before)

    via_matrix = copy.deepcopy(base2ee)
    end_matrix_xyzrpy = copy.deepcopy(current_xyzrpw)

    # 로봇에 맞게 좌표 변경 이전에 몇센치 뒤에 있다가 빠지는 지 
    via_matrix_xyzrpy = interpolate_xyzrpy(current_xyzrpw, base2target_xyzrpy, alpha=0.5)

    if int(arm_flag) == int(NetProto_Arm.LEFT): # left
        print("[LEFT]")
        via_position = np.array([0.4 , -0.25, base2target_xyzrpy[2]+0.1]) # endeffector 좌표계 기준으로 생각해야
        end_position = np.array([0.3 , 0.0, 0.15])
        if object_name == "the red box.":
            end_position = np.array([0.3 , 0.0, 0.20])
        elif object_name == "the red can." or object_name == "the white can.":
            end_position = np.array([0.3 , 0.0, 0.14])
        # 0.39706217 -0.08047105  0.40784926
    else:
        print("[RIGHT]")
        via_position = np.array([0.4 , 0.25 , base2target_xyzrpy[2]+0.1])
        end_position = np.array([0.3 , 0.0 , 0.15])
        if object_name == "the red box.":
            end_position = np.array([0.3 , 0.0, 0.20])
        elif object_name == "the red can." or object_name == "the white can.":
            end_position = np.array([0.3 , 0.0, 0.14])
    
    via_matrix_xyzrpy[:3] = via_position
    end_matrix_xyzrpy[:3] = end_position

    print("via_matrix ",via_matrix)
    print("base2target_before ",base2target_before)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    target_frame.transform(base2target_before)
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    target_sphere.paint_uniform_color([0, 0, 1]) # blue
    target_sphere.translate(base2target_before[:3, 3])

    via_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    via_frame.transform(via_matrix)
    via_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    via_sphere.paint_uniform_color([0, 0, 1]) # blue
    via_sphere.translate(via_matrix[:3, 3])

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


    # o3d.visualization.draw_geometries([base_frame,ee_frame, cam_frame, target_frame,via_frame,ee_sphere,base_sphere,cam_sphere, target_sphere, via_sphere])



    via_matrix = mat_to_xyzrpw(via_matrix)
    base2target_before = mat_to_xyzrpw(base2target_before)
    base2target = mat_to_xyzrpw(base2target)
    base2ee = mat_to_xyzrpw(base2ee)
    print("via_matrix_xyzrpy ", via_matrix_xyzrpy)
    print(" base2target_before", base2target_before)
    print("base2target ", base2target)
    print("base2ee ", base2ee)

    via_matrix_xyzrpy = offset_y(via_matrix_xyzrpy, arm_flag)
    base2target_before_xyzrpy = offset_y(base2target_before_xyzrpy,arm_flag)
    base2target_xyzrpy = offset_y(base2target_xyzrpy,arm_flag)
    end_matrix_xyzrpy = offset_y(end_matrix_xyzrpy, arm_flag)

    # Arm movements
    # via matrix로 이동





    motion_time = compute_motion_time(current_xyzrpw, via_matrix_xyzrpy)
    send_and_wait_for_motion(via_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)
    
    # before matrix로 이동
    motion_time = compute_motion_time(via_matrix_xyzrpy, base2target_before_xyzrpy)
    send_and_wait_for_motion(base2target_before_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline, motion_time)
    
    # target으로 이동
    motion_time = compute_motion_time(base2target_before_xyzrpy, base2target_xyzrpy)
    send_and_wait_for_motion(base2target_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline, motion_time)

    # # Gripper close
    wait_for_gripper_action(socket_control, NetProto.GRIPPER_CLOSE, ARM_flag, pipeline)

    # before matrix로 이동
    motion_time = compute_motion_time(base2target_xyzrpy, base2target_before_xyzrpy)
    send_and_wait_for_motion(base2target_before_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline, motion_time)

    # via matrix로 이동
    motion_time = compute_motion_time(base2target_before_xyzrpy, via_matrix_xyzrpy)
    send_and_wait_for_motion(via_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    # # current로 이동
    # motion_time = compute_motion_time(via_matrix_xyzrpy, current_xyzrpw)
    # send_and_wait_for_motion(current_xyzrpw, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    # end matrix로 이동
    motion_time = compute_motion_time(via_matrix_xyzrpy, end_matrix_xyzrpy)
    send_and_wait_for_motion(end_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    wait_for_gripper_action(socket_control, NetProto.GRIPPER_OPEN, ARM_flag, pipeline)

    # current로 이동
    motion_time = compute_motion_time(end_matrix_xyzrpy, current_xyzrpw)
    send_and_wait_for_motion(current_xyzrpw, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)


    # end_matrix_xyzrpy

    # wait_for_gripper_action(socket_control, NetProto.GRIPPER_OPEN, ARM_flag, pipeline)

    out.release()
    cv2.destroyAllWindows()
    # pipeline.stop()