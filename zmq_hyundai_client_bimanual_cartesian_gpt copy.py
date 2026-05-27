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

# 변수 정의
data_dir = "/home/vision/packages/FoundationPose/result"

CALIB_PATH_LEFT = "/home/vision/packages/calibration/data/data_1030_left/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"
CALIB_PATH_RIGHT = "/home/vision/packages/calibration/data/data_1030_right/no_depth/hand_eye_v1_c2b_e2c_nominal_mean.yaml"

# gipper 상태 저장 / 열려있는 상태에서 시작 
gripper_left = True
gripper_right = True

# Camera congifuration
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

# ZMQ Definition
context_control = zmq.Context()
socket_control = context_control.socket(zmq.REQ)
socket_control.connect("tcp://161.122.114.39:5555")



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
    GRIPPER = 7
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
    socket_control.send(flag + q.tobytes() + time.tobytes())

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


def wait_for_gripper_action(socket_control, flag_code, arm_flag, pipeline, gripper_left, gripper_right):
    flag = bytes([int(flag_code)])
    # true(1)면 열기/ false(0)면 닫기
    grip_flag_left = bytes([gripper_left])
    grip_flag_right = bytes([gripper_right])

    socket_control.send(flag+grip_flag_left+grip_flag_right)

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
    if (arm_flag) == 'left': # left
        print("LEFT")
        xyzrpy[1] = xyzrpy[1] + 0.3
        # 0.39706217 -0.08047105  0.40784926
    elif arm_flag == 'right':
        print("RIGHT")
        xyzrpy[1] = xyzrpy[1] - 0.3
    else: 
        print("❌ ARM Selection Error ❌")
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
def compute_motion_time(current_xyzrpy, target_xyzrpy, arm_flag= None,
                        max_linear_speed=0.08, max_angular_speed=0.5):
    """
    XYZRPY 두 pose를 이용해 motion time 계산
    :param current_xyzrpy: 현재 pose [x, y, z, roll, pitch, yaw] (m, rad)
    :param target_xyzrpy: 목표 pose [x, y, z, roll, pitch, yaw] (m, rad)
    :param max_linear_speed: 최대 선속도 [m/s]
    :param max_angular_speed: 최대 각속도 [rad/s]
    :return: motion_time [s]
    """
    def single_arm_time(curr, tgt):
        # --- 1) 위치 차이 ---
        pos_diff = np.linalg.norm(tgt[:3] - curr[:3])

        # --- 2) 회전 차이 ---
        R_current = R.from_euler('xyz', curr[3:], degrees=False)
        R_target = R.from_euler('xyz', tgt[3:], degrees=False)
        rot_diff = (R_target * R_current.inv()).magnitude()

        # --- 3) motion time ---
        time_pos = pos_diff / max_linear_speed
        time_rot = rot_diff / max_angular_speed
        return max(time_pos, time_rot)
    
    min_motion_time = 3.0
    # ---------- 분기 ----------
    if arm_flag == "left":
        t = single_arm_time(current_xyzrpy[:6], target_xyzrpy[:6])
        return max(t, min_motion_time)

    elif arm_flag == "right":
        t = single_arm_time(current_xyzrpy[6:], target_xyzrpy[6:])
        return max(t, min_motion_time)

    elif arm_flag == "both":
        left_time = single_arm_time(current_xyzrpy[:6], target_xyzrpy[:6])
        right_time = single_arm_time(current_xyzrpy[6:], target_xyzrpy[6:])
        t = max(left_time, right_time)
        return max(t, min_motion_time)

    else:
        raise ValueError(f"Invalid arm_flag: {arm_flag}")



def align_gripper(base2target):
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
        if dot > -0.1:
            print("그리퍼가 y 바깥을 바라보도록!")
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
    else:
        print("그리퍼 아래로 잘 향해 있음. dot_down : ",dot_down)
    target_z = base2target[:3,2]
    target_y = base2target[:3,1]
    dot_foward = np.dot(target_z, forward)
    print("dot forward ",dot_foward)
    if dot_foward > 0.7: # 그리퍼가 z축 앞으로 향해 있을 떄
        print("그리퍼 z 앞으로 향하고")
        dot = np.dot(target_y,base_z)
        if dot > -0.1:
            print("그리퍼가 y 위를 바라보도록!")
            correction = np.eye(4)
            correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
            base2target = base2target @ correction
    else:
        print("그리퍼 앞으로 잘 향해 있음. dot_down : ",dot_foward)

    # correction = np.eye(4)
    # correction[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()   
    # base2target = base2target @ correction
    
    # 이거 왜하지?
    correction = np.eye(4)
    correction[:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix() 
    base2target = base2target @ correction
    
    return base2target


def execute_single_arm_grasp(task,next_skill):
    global gripper_left, gripper_right
    arm = None
    if 'left' in task['skill']:
        arm = 'left'
    elif 'right' in task['skill']:
        arm = 'right'
    else:
        print("❌ ARM Selection Error ❌")

    grasp_info = data_load_grasp(data_dir, object_name, arm)

    cam2target = grasp_info['transformation']

    cam2target = np.array(cam2target)

    # 로봇에 맞게 좌표 변경 endeffector 끝으로 변경 
    half_depth_vector = np.array([0 , 0, 0.1034])
    cam2target[:3, 3] = cam2target[:3, 3] + cam2target[:3, :3] @ half_depth_vector


    print("==================================================")
    print('cam2target:', cam2target)

    arm_flag = grasp_info['arm']
    print("arm_flag ",arm_flag)

            
    flag = bytes([int(NetProto.REQ_XYZROLLPITCHYAW)])
    socket_control.send(flag)
    data = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

    if len(data) != 12:
        print("⚠ Warning: received data length =", len(data))

    current_xyzrpw_l = data[:6]      # 첫 6개
    current_xyzrpw_r = data[6:12]   # 다음 6개

    print("LEFT  xyzrpw =", current_xyzrpw_l)
    print("RIGHT xyzrpw =", current_xyzrpw_r)
    

    if (arm_flag) == 'left': 
        ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
        file_path = CALIB_PATH_LEFT
        current_xyzrpw = current_xyzrpw_l
    elif (arm_flag) == 'right': 
        ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
        file_path = CALIB_PATH_RIGHT
        current_xyzrpw = current_xyzrpw_r

    calib_info = data_load_calib(file_path)
    cam2base = np.array(calib_info['c2b'])
    cam2base[:3, 3] /= 1000.0


    print("cam2target ", cam2target)
    print("cam2base ", cam2base)


    base2target = cam2base @ cam2target 
    base2target = align_gripper(base2target)
    half_depth_vector = np.array([0 , 0, -0.05])
    base2target_before = copy.deepcopy(base2target)
    base2target_before[:3, 3] = base2target_before[:3, 3] + base2target_before[:3, :3] @ half_depth_vector
    base2ee = xyzrpw_to_mat(current_xyzrpw)
    base2target_xyzrpy = mat_to_xyzrpw(base2target)
    base2target_before_xyzrpy = mat_to_xyzrpw(base2target_before)
    via_matrix = copy.deepcopy(base2ee)
    end_matrix_xyzrpy = copy.deepcopy(current_xyzrpw)

    if arm == 'left':
        y_nom = base2target_xyzrpy[1] + 0.2
        y = np.clip(y_nom, -0.1, 0.2)

        
    elif arm == 'right':
        y_nom = base2target_xyzrpy[1] - 0.2
        y = np.clip(y_nom, -0.2, 0.1)
    
    base2target_before_xyzrpy[2] = base2target_before_xyzrpy[2] + 0.03 # 아래서부터 위로 잡는 경우에 잡고 뒤로 빠지면서 물체가 선반 바닥과 충돌해서 콜리젼뜸

    # 로봇에 맞게 좌표 변경 이전에 몇센치 뒤에 있다가 빠지는 지 
    via_matrix_xyzrpy = interpolate_xyzrpy(current_xyzrpw, base2target_xyzrpy, alpha=0.5)
    if (arm_flag) == "left": # left
        print("[LEFT]")
        via_position = np.array([via_matrix_xyzrpy[0] , y, base2target_xyzrpy[2]+0.05]) # endeffector 좌표계 기준으로 생각해야
        # 0.39706217 -0.08047105  0.40784926
    elif arm_flag == 'right':
        print("[RIGHT]")
        via_position = np.array([via_matrix_xyzrpy[0] , y, base2target_xyzrpy[2]+0.05])
    else:
        print("❌ ARM Selection Error ❌")
        
    via_matrix_xyzrpy[:3] = via_position
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

    # packing
    current_xyzrpw = np.concatenate([current_xyzrpw_l, current_xyzrpw_r])
    if arm_flag == "left":
        base2target_xyzrpy = np.concatenate([base2target_xyzrpy, current_xyzrpw_r])
        base2target_before_xyzrpy = np.concatenate([base2target_before_xyzrpy, current_xyzrpw_r])
        via_matrix_xyzrpy = np.concatenate([via_matrix_xyzrpy, current_xyzrpw_r])
        gripper_left = not gripper_left
    elif arm_flag == "right":
        base2target_xyzrpy = np.concatenate([current_xyzrpw_l, base2target_xyzrpy])
        base2target_before_xyzrpy = np.concatenate([current_xyzrpw_l, base2target_before_xyzrpy])
        via_matrix_xyzrpy = np.concatenate([current_xyzrpw_l, via_matrix_xyzrpy])
        gripper_right = not gripper_right

    # Arm movements
    # via matrix로 이동
    motion_time = compute_motion_time(current_xyzrpw, via_matrix_xyzrpy, arm_flag)
    send_and_wait_for_motion(via_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    # before matrix로 이동
    motion_time = compute_motion_time(via_matrix_xyzrpy, base2target_before_xyzrpy,arm_flag)
    send_and_wait_for_motion(base2target_before_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline, motion_time)

    # target으로 이동
    motion_time = compute_motion_time(base2target_before_xyzrpy, base2target_xyzrpy,arm_flag)
    send_and_wait_for_motion(base2target_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline, motion_time)

    # # Gripper close
    wait_for_gripper_action(socket_control, NetProto.GRIPPER, arm_flag, pipeline, gripper_left, gripper_right)

    # before matrix로 이동
    motion_time = compute_motion_time(base2target_xyzrpy, base2target_before_xyzrpy,arm_flag)
    send_and_wait_for_motion(base2target_before_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline, motion_time)

    # via matrix로 이동
    motion_time = compute_motion_time(base2target_before_xyzrpy, via_matrix_xyzrpy,arm_flag)
    send_and_wait_for_motion(via_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    # current로 이동
    if True: #next_skill is None or 'place' not in next_skill:
        motion_time = compute_motion_time(via_matrix_xyzrpy, current_xyzrpw, arm_flag)
        send_and_wait_for_motion(current_xyzrpw, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)


def execute_bimanual_grasp(task ,next_skill):
    global gripper_left, gripper_right
    object_name_l  = task['object_name']['left']
    object_name_r = task['object_name']['right']

    grasp_info_l = data_load_grasp(data_dir, object_name_l, 'left')
    grasp_info_r = data_load_grasp(data_dir, object_name_r, 'right')

    cam2target_l = grasp_info_l['transformation']
    cam2target_r = grasp_info_r['transformation']

    cam2target_l = np.array(cam2target_l)
    cam2target_r = np.array(cam2target_r)

    # 로봇에 맞게 좌표 변경 endeffector 끝으로 변경 
    half_depth_vector = np.array([0 , 0, 0.1034])
    cam2target_l[:3, 3] = cam2target_l[:3, 3] + cam2target_l[:3, :3] @ half_depth_vector
    cam2target_r[:3, 3] = cam2target_r[:3, 3] + cam2target_r[:3, :3] @ half_depth_vector


    print("==================================================")
    print('cam2target_l:', cam2target_l)
    print('cam2target_r:', cam2target_r)

            
    flag = bytes([int(NetProto.REQ_XYZROLLPITCHYAW)]) # 양팔 값 받아옴
    socket_control.send(flag)
    data = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

    if len(data) != 12:
        print("⚠ Warning: received data length =", len(data))

    current_xyzrpw_l = data[:6]      # 첫 6개
    current_xyzrpw_r = data[6:12]   # 다음 6개

    print("LEFT  xyzrpw =", current_xyzrpw_l)
    print("RIGHT xyzrpw =", current_xyzrpw_r)


    calib_info_l = data_load_calib(CALIB_PATH_LEFT)
    calib_info_r = data_load_calib(CALIB_PATH_RIGHT)

    cam2base_l = np.array(calib_info_l['c2b'])
    cam2base_l[:3, 3] /= 1000.0
    cam2base_r = np.array(calib_info_r['c2b'])
    cam2base_r[:3, 3] /= 1000.0


    base2target_l = cam2base_l @ cam2target_l 
    base2target_l = align_gripper(base2target_l)
    base2target_r = cam2base_r @ cam2target_r 
    base2target_r = align_gripper(base2target_r)


    half_depth_vector = np.array([0 , 0, -0.05])
    base2target_before_l = copy.deepcopy(base2target_l)
    base2target_before_l[:3, 3] = base2target_before_l[:3, 3] + base2target_before_l[:3, :3] @ half_depth_vector
    base2target_before_r = copy.deepcopy(base2target_r)
    base2target_before_r[:3, 3] = base2target_before_r[:3, 3] + base2target_before_r[:3, :3] @ half_depth_vector

    base2ee_l = xyzrpw_to_mat(current_xyzrpw_l)
    base2target_xyzrpy_l = mat_to_xyzrpw(base2target_l)
    base2target_before_xyzrpy_l = mat_to_xyzrpw(base2target_before_l)
    base2ee_r = xyzrpw_to_mat(current_xyzrpw_r)
    base2target_xyzrpy_r = mat_to_xyzrpw(base2target_r)
    base2target_before_xyzrpy_r = mat_to_xyzrpw(base2target_before_r)

    base2target_before_xyzrpy_l[2] = base2target_before_xyzrpy_l[2] + 0.03
    base2target_before_xyzrpy_r[2] = base2target_before_xyzrpy_r[2] + 0.03

    via_matrix_l = copy.deepcopy(base2ee_l)
    via_matrix_r = copy.deepcopy(base2ee_r)

    # 로봇에 맞게 좌표 변경 이전에 몇센치 뒤에 있다가 빠지는 지 
    via_matrix_xyzrpy_l = interpolate_xyzrpy(current_xyzrpw_l, base2target_xyzrpy_l, alpha=0.5)
    via_matrix_xyzrpy_r = interpolate_xyzrpy(current_xyzrpw_r, base2target_xyzrpy_r, alpha=0.5)


    # y offset 설정 위해 기본 오프셋 (물체 기준으로 따라갈 때)
    yL_nom = base2target_xyzrpy_l[1] + 0.1
    yR_nom = base2target_xyzrpy_r[1] - 0.1

    # gap = abs(yL_nom - yR_nom)
    # mid = 0.5 * (yL_nom + yR_nom)
    # print("gap ", gap)
    # # 임계값/지정값 (원하는 대로 튜닝)
    # gap_min = 0.18          # 이보다 가까우면 "너무 붙음"
    # gap_max = 0.45          # 이보다 멀면 "너무 멈"
    # sep_when_close = 0.22   # 너무 붙으면 이 간격으로 벌려서 감
    # sep_when_far   = 0.35   # 너무 멀면 이 간격으로 좁혀서 감

    # if gap < gap_min:
    #     print("너무 가까움 -> 강제로 벌리기")
    #     # 너무 가까움 -> 강제로 벌리기
    #     sep = sep_when_close
    #     yL = mid + sep / 2
    #     yR = mid - sep / 2

    # elif gap > gap_max:
    #     print("너무 멂 -> 강제로 좁히기")
    #     # 너무 멂 -> 강제로 좁히기 (원하면 여기서도 "어떤 값으로 고정" 가능)
    #     sep = sep_when_far
    #     yL = mid + sep / 2
    #     yR = mid - sep / 2

    # else:
    #     print("적당함 -> 물체 위 위치를 그대로 따름")
    #     # 적당함 -> 물체 위 위치를 그대로 따름
    #     yL = yL_nom
    #     yR = yR_nom

    
    yL = np.clip(yL_nom, -0.1, 0.2)
    yR = np.clip(yR_nom, -0.2, 0.1)

    print("yL_nom : ", yL_nom)
    print("yR_nom : ", yR_nom)
    print("yL : ", yL)
    print("yR : ", yR)

    # exit()

    via_position_l = np.array([via_matrix_xyzrpy_l[0] , yL , base2target_xyzrpy_l[2]]) # endeffector 좌표계 기준으로 생각해야
    via_position_r = np.array([via_matrix_xyzrpy_r[0] , yR , base2target_xyzrpy_r[2]])


    via_matrix_xyzrpy_l[:3] = via_position_l
    via_matrix_xyzrpy_r[:3] = via_position_r


    
    via_matrix_l = mat_to_xyzrpw(via_matrix_l)
    base2target_before_l = mat_to_xyzrpw(base2target_before_l)
    base2target_l = mat_to_xyzrpw(base2target_l)
    base2ee_l = mat_to_xyzrpw(base2ee_l)
    via_matrix_r = mat_to_xyzrpw(via_matrix_r)
    base2target_before_r = mat_to_xyzrpw(base2target_before_r)
    base2target_r = mat_to_xyzrpw(base2target_r)
    base2ee_r = mat_to_xyzrpw(base2ee_r)


    via_matrix_xyzrpy_l = offset_y(via_matrix_xyzrpy_l, 'left')
    base2target_before_xyzrpy_l = offset_y(base2target_before_xyzrpy_l,'left')
    base2target_xyzrpy_l = offset_y(base2target_xyzrpy_l,'left')
    via_matrix_xyzrpy_r = offset_y(via_matrix_xyzrpy_r, 'right')
    base2target_before_xyzrpy_r = offset_y(base2target_before_xyzrpy_r,'right')
    base2target_xyzrpy_r = offset_y(base2target_xyzrpy_r,'right')
    
    # packing
    current_xyzrpw = np.concatenate([current_xyzrpw_l, current_xyzrpw_r])
    base2target_xyzrpy = np.concatenate([base2target_xyzrpy_l, base2target_xyzrpy_r])
    base2target_before_xyzrpy = np.concatenate([base2target_before_xyzrpy_l, base2target_before_xyzrpy_r])
    via_matrix_xyzrpy = np.concatenate([via_matrix_xyzrpy_l, via_matrix_xyzrpy_r])
    gripper_left = not gripper_left
    gripper_right = not gripper_right

    # Arm movements
    # via matrix로 이동
    motion_time = compute_motion_time(current_xyzrpw, via_matrix_xyzrpy, 'both')
    send_and_wait_for_motion(via_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline,motion_time)

    # before matrix로 이동
    motion_time = compute_motion_time(via_matrix_xyzrpy, base2target_before_xyzrpy,'both')
    send_and_wait_for_motion(base2target_before_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline, motion_time)

    # target으로 이동
    motion_time = compute_motion_time(base2target_before_xyzrpy, base2target_xyzrpy,'both')
    send_and_wait_for_motion(base2target_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline, motion_time)

    # # Gripper close
    wait_for_gripper_action(socket_control, NetProto.GRIPPER, 'both', pipeline, gripper_left, gripper_right)

    # before matrix로 이동
    motion_time = compute_motion_time(base2target_xyzrpy, base2target_before_xyzrpy,'both')
    send_and_wait_for_motion(base2target_before_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline, motion_time)

    # # via matrix로 이동
    motion_time = compute_motion_time(base2target_before_xyzrpy, via_matrix_xyzrpy,'both')
    send_and_wait_for_motion(via_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline,motion_time)

    # current로 이동
        # current로 이동
    if True: # next_skill is None or 'place' not in next_skill:
        motion_time = compute_motion_time(via_matrix_xyzrpy, current_xyzrpw, 'both')
        send_and_wait_for_motion(current_xyzrpw, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline,motion_time)


def execute_single_arm_place(task):
    global gripper_left, gripper_right
    arm = None
    if 'left' in task['skill']:
        arm = 'left'
    elif 'right' in task['skill']:
        arm = 'right'
    else:
        print("❌ ARM Selection Error ❌")

    print("arm_flag ",arm)

    flag = bytes([int(NetProto.REQ_XYZROLLPITCHYAW)])
    socket_control.send(flag)
    data = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)
    if len(data) != 12:
        print("⚠ Warning: received data length =", len(data))
    current_xyzrpw_l = data[:6]      # 첫 6개
    current_xyzrpw_r = data[6:12]   # 다음 6개

    print("LEFT  xyzrpw =", current_xyzrpw_l)
    print("RIGHT xyzrpw =", current_xyzrpw_r)

    if (arm) == 'left': 
        ARM_flag = bytes([int(NetProto_Arm.LEFT)]) 
        file_path = CALIB_PATH_LEFT
        current_xyzrpw = current_xyzrpw_l
    elif (arm) == 'right': 
        ARM_flag = bytes([int(NetProto_Arm.RIGHT)]) 
        file_path = CALIB_PATH_RIGHT
        current_xyzrpw = current_xyzrpw_r

    calib_info = data_load_calib(file_path)
    cam2base = np.array(calib_info['c2b'])
    cam2base[:3, 3] /= 1000.0
    base2ee = xyzrpw_to_mat(current_xyzrpw)
    end_matrix_xyzrpy = copy.deepcopy(current_xyzrpw)
    end_matrix_xyzrpy_rot = copy.deepcopy(current_xyzrpw) # rotation 만 먼저 

    if (arm) == 'left': # left
        print("[LEFT]")
        end_position =  np.array([0.4 , 0.0, 0.15]) #np.array([0.6 , 0.40, 0.15])
        if object_name == "the red box.":
            end_position = np.array([0.45 , 0.0, 0.20])
        elif object_name == "the red can." or object_name == "the white can.":
            end_position = np.array([0.45 , 0.0, 0.14])
        # 0.39706217 -0.08047105  0.40784926
    elif (arm) == 'right': # right
        print("[RIGHT]")
        end_position = np.array([0.4 , 0.0, 0.15]) # np.array([0.6 , -0.40 , 0.15])
        if object_name == "the red box.":
            end_position = np.array([0.45 , 0.0, 0.20])
        elif object_name == "the red can." or object_name == "the white can.":
            end_position = np.array([0.45 , 0.0, 0.14])
    else:
        print("❌ ARM Selection Error ❌")
    
    end_rotation = np.array([-3.13, -0.00,  0.00])
    
    end_matrix_xyzrpy[:3] = end_position
    # end_matrix_xyzrpy[3:] = end_rotation
    # end_matrix_xyzrpy_rot[3:] = end_rotation

    
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
    
    end_matrix_xyzrpy = offset_y(end_matrix_xyzrpy, arm)
    # end_matrix_xyzrpy_rot = offset_y(end_matrix_xyzrpy_rot, arm)

    print("end_matrix_xyzrpy ", end_matrix_xyzrpy)
    # packing
    if arm == "left":
        end_matrix_xyzrpy = np.concatenate([end_matrix_xyzrpy, current_xyzrpw_r])
        # end_matrix_xyzrpy_rot = np.concatenate([end_matrix_xyzrpy_rot, current_xyzrpw_r])
        gripper_left = not gripper_left
    elif arm == "right":
        end_matrix_xyzrpy = np.concatenate([current_xyzrpw_l, end_matrix_xyzrpy])
        # end_matrix_xyzrpy_rot = np.concatenate([current_xyzrpw_l, end_matrix_xyzrpy_rot])
        gripper_right = not gripper_right


    current_xyzrpw = np.concatenate([current_xyzrpw_l, current_xyzrpw_r])

    # # rotation 먼저 하고
    # motion_time = compute_motion_time(current_xyzrpw, end_matrix_xyzrpy_rot, arm)
    # send_and_wait_for_motion(end_matrix_xyzrpy_rot, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    # end matrix로 이동
    motion_time = compute_motion_time(current_xyzrpw, end_matrix_xyzrpy, arm)
    send_and_wait_for_motion(end_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)

    # 그리퍼 풀고
    wait_for_gripper_action(socket_control, NetProto.GRIPPER, arm, pipeline, gripper_left, gripper_right)

    # rotation 하고 
    # end_matrix_xyzrpy_rot[:3] = end_position
    # motion_time = compute_motion_time(end_matrix_xyzrpy, end_matrix_xyzrpy_rot, arm)
    # send_and_wait_for_motion(end_matrix_xyzrpy_rot, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)
    

    # current로 이동
    motion_time = compute_motion_time(end_matrix_xyzrpy, current_xyzrpw, arm)
    send_and_wait_for_motion(current_xyzrpw, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, ARM_flag, pipeline,motion_time)



def execute_bimanual_place(task):
    global gripper_left, gripper_right
    arm = 'both'

    flag = bytes([int(NetProto.REQ_XYZROLLPITCHYAW)])
    socket_control.send(flag)
    data = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)
    if len(data) != 12:
        print("⚠ Warning: received data length =", len(data))
    current_xyzrpw_l = data[:6]      # 첫 6개
    current_xyzrpw_r = data[6:12]   # 다음 6개

    print("LEFT  xyzrpw =", current_xyzrpw_l)
    print("RIGHT xyzrpw =", current_xyzrpw_r)


    calib_info_l = data_load_calib(CALIB_PATH_LEFT)
    cam2base_l = np.array(calib_info_l['c2b'])
    cam2base_l[:3, 3] /= 1000.0
    end_matrix_xyzrpy_l = copy.deepcopy(current_xyzrpw_l)

    calib_info_r = data_load_calib(CALIB_PATH_RIGHT)
    cam2base_r = np.array(calib_info_r['c2b'])
    cam2base_r[:3, 3] /= 1000.0
    end_matrix_xyzrpy_r = copy.deepcopy(current_xyzrpw_r)

    end_position_l = np.array([0.4 , 0.0, 0.15]) #np.array([0.45 , 0.0, 0.15])
    end_rotation = np.array([-3.13, -0.00,  0.00])
    # if object_name == "the red box.":
    #     end_position = np.array([0.3 , 0.0, 0.20])
    # elif object_name == "the red can." or object_name == "the white can.":
    #     end_position = np.array([0.3 , 0.0, 0.14])
        
    end_position_r = np.array([0.4 , 0.0, 0.15]) #np.array([0.45 , 0.0 , 0.15])
    # if object_name == "the red box.":
    #     end_position = np.array([0.3 , 0.0, 0.20])
    # elif object_name == "the red can." or object_name == "the white can.":
    #     end_position = np.array([0.3 , 0.0, 0.14])

        
    end_matrix_xyzrpy_l[:3] = end_position_l
    # end_matrix_xyzrpy_l[3:] = end_rotation
    
    end_matrix_xyzrpy_r[:3] = end_position_r
    # end_matrix_xyzrpy_r[3:] = end_rotation

    end_matrix_xyzrpy_l = offset_y(end_matrix_xyzrpy_l, 'left')
    end_matrix_xyzrpy_r = offset_y(end_matrix_xyzrpy_r, 'right')

    # packing
    end_matrix_xyzrpy = np.concatenate([end_matrix_xyzrpy_l, end_matrix_xyzrpy_r])
    gripper_left = not gripper_left
    gripper_right = not gripper_right

    current_xyzrpw = np.concatenate([current_xyzrpw_l, current_xyzrpw_r])

    # end matrix로 이동
    motion_time = compute_motion_time(current_xyzrpw, end_matrix_xyzrpy, arm)
    send_and_wait_for_motion(end_matrix_xyzrpy, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline,motion_time)

    # 그리퍼 풀고
    wait_for_gripper_action(socket_control, NetProto.GRIPPER, arm, pipeline, gripper_left, gripper_right)

    # current로 이동
    motion_time = compute_motion_time(end_matrix_xyzrpy, current_xyzrpw, arm)
    send_and_wait_for_motion(current_xyzrpw, socket_control, NetProto.MOVE_ARM_BY_XYZRPY, 'both', pipeline,motion_time)


if __name__ == '__main__':

    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--text_prompt', type=str , help='Model checkpoint path', default=None, required=False)
    # cfgs = parser.parse_args()
    # object_name = cfgs.text_prompt

    print("==== ✅ Check GPT Output ===")
    with open("/home/vision/packages/FoundationPose/result/gpt_planning_output.json", "r") as f:
      data = json.load(f)

    planner = (data["Plan"])
    print("Task Num :", len(planner))

    


    

    
    count = 0

    # data_dir = "/home/vision/packages/Scale-Balanced-Grasp"
    



    for i, task in enumerate(planner):
        print(task)
        object_name = task['object_name']
        if 'grasp' in task['skill']:
            print(task['skill'])

            next_task = planner[i+1] if i + 1 < len(planner) else None
            next_skill = next_task['skill'] if next_task else None

            if 'left' in task['skill'] or 'right' in task['skill']:
                execute_single_arm_grasp(task, next_skill)
            elif 'both' in task['skill']:
                execute_bimanual_grasp(task , next_skill)
            else:
                print("❌ ARM Selection Error ❌")

            
            
        
        elif 'place' in task['skill']:
            print(task['skill'])

            arm = None
            if 'left' in task['skill'] or 'right' in task['skill']:
                execute_single_arm_place(task)

            elif 'both' in task['skill']:
                arm = 'both'
                execute_bimanual_place(task)
            else:
                print("❌ ARM Selection Error ❌")



            

        
        else:
            print("❌ Something was wrong! ❌")
    out.release()
    cv2.destroyAllWindows()
    # pipeline.stop()