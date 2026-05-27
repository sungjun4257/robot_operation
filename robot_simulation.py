import numpy as np
import cv2

import zmq
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

# from utils.DH_Panda import Panda

from dh_utils.DH_Panda import Panda

from spatialmath import SE3

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
import yaml


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR,'contact_grasp'))
from contact_grasp.mesh_utils import *



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

def matrix2xyzrpy(matrix):
    x, y, z = matrix[:3, 3]
    roll, pitch, yaw = R.from_matrix(matrix[:3,:3]).as_euler('xyz')

    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)

# param = np.array(np.load('./dh_utils/param/dh_param_1024.npy'))
# robot = Panda(param)

# ee2cam = np.load('./dh_utils/param/ee2cam_1024.npy')

# pipeline = rs.pipeline()
# config = rs.config()

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# profile = pipeline.start(config)

# align_to = rs.stream.color
# align = rs.align(align_to)

# color_stream = profile.get_stream(rs.stream.color)
# intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
#                         [0, intrinsics.fy, intrinsics.ppy],
#                         [0, 0, 1]]).astype(np.float32)

# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://localhost:1111")

# context_control = zmq.Context()
# socket_control = context_control.socket(zmq.REQ)
# socket_control.connect("tcp://192.168.100.150:5555")

# is_selecting = False
# ref_pt = []


def data_load_image(data_dir="/home/vision/packages/FoundationPose/result"):
    
    color_rs = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths_rs = np.array(Image.open(os.path.join(data_dir, 'depth.png'))) / 1000.0

    fx, fy = 612.6182250976562, 612.7216796875
    cx, cy = 318.95758056640625, 240.05343627929688

    # get point cloud
    xmap, ymap = np.arange(depths_rs.shape[1]), np.arange(depths_rs.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths_rs 
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    mask = (points_z > 0) & (points_z < 100)
    points_rs = np.stack([points_x, points_y, points_z], axis=-1)

    points_rs = points_rs[mask].astype(np.float32)
    color_rs = color_rs[mask].astype(np.float32)
    

    cloud_scene = o3d.geometry.PointCloud()
    cloud_scene.points = o3d.utility.Vector3dVector(points_rs)
    cloud_scene.colors = o3d.utility.Vector3dVector(color_rs)
    cloud_scene,_ = cloud_scene.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)


    return cloud_scene

def data_load_calib(file_path = None):
    # file_path = os.path.join(data_dir, "calibration", "hand_eye_v1_c2b_calib_mean.yaml")
    # file_path = os.path.join(data_dir, "calibration", "hand_eye_v1_whole_calib_mean.yaml")
    # file_path = os.path.join(data_dir, "calibration","hand_eye_v1_whole_nominal_mean.yaml" )
    # file_path = os.path.join(data_dir, "calibration/0423/right","hand_eye_v1_c2b_e2c_nominal_mean.yaml" )
    # file_path = os.path.join(data_dir, "calibration/0423/right","hand_eye_v1_c2b_e2c_single_nominal_mean.yaml" )
    # file_path = os.path.join(data_dir, "calibration/0423/right","hand_eye_v1_whole_nominal_mean.yaml" )

    # file_path = os.path.join(data_dir, "calibration/0423/left","hand_eye_v1_c2b_e2c_nominal_mean.yaml" )
    # file_path = os.path.join(data_dir, "calibration/0423/left","hand_eye_v1_c2b_e2c_single_nominal_mean.yaml" )
    # file_path = os.path.join(data_dir, "calibration/0423/left","hand_eye_v1_c2b_nominal_mean.yaml" )


    with open(file_path, 'r') as f:
        data = yaml.load(f, yaml.FullLoader)

    return data

    

def data_load_cad(data_dir="/home/vision/packages/FoundationPose/result"):
    file_path = os.path.join(data_dir, 'config.json')
    with open(file_path, "rb") as f:
        config_data = json.load(f)

    mesh = o3d.io.read_triangle_mesh(config_data["mesh_path"])
    # tri_mesh = trimesh.load(config_data["mesh_path"])
    cloud = mesh.sample_points_uniformly(number_of_points=50000)
    # 🔹 Normal Vector 계산 (KNN 기반)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # 🔹 Normal 방향 정렬 (Tangent Plane 기준)
    cloud.orient_normals_consistent_tangent_plane(100)

    cloud_color = np.full_like(cloud.points, [1.0, 1.0, 0.6])  # Default gray color
    cloud.colors = o3d.utility.Vector3dVector(cloud_color.astype(np.float32))

    pose = np.load(os.path.join(data_dir,"output_pose.npy")).astype(np.float32)
    cloud.transform(pose)


    return cloud

def data_load_grasp(data_dir):
    
    # 파일 경로
    file_path = os.path.join(data_dir,"best_grasp_panda.json")

    # JSON 파일 읽기
    with open(file_path, 'r') as json_file:
        grasp_data = json.load(json_file)

    print(grasp_data)

    return grasp_data

def data_load_grasp(data_dir, object_name, arm):
    '''
    data_dir    : base directoroy 
    object_name : object name
    arm         : name
    object name에 해당 팔에 맞는 grasp_data load
    '''
    data_dir = os.path.join(data_dir, object_name)
    # 파일 경로
    file_path = os.path.join(data_dir,"best_grasp_panda.json")

    # JSON 파일 읽기
    with open(file_path, 'r') as json_file:
        grasp_datas = json.load(json_file)

    for data in grasp_datas['grasps']:
        if data['arm'] == arm:
            grasp_data = data
            break

    print(grasp_data)

    return grasp_data


# def click_and_crop(event, x, y, flags, param):
#     global is_selecting
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         is_selecting = True
#         ref_pt.append([x, y])
#     # check to see if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates and indicate that
#         # the cropping operation is finished
#         if len(ref_pt) == 2:
#             is_selecting = False
#             ref_pt.append([x, y])
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if is_selecting == True:
#             if len(ref_pt) == 1:
#                 ref_pt.append([x, y])
#             elif len(ref_pt) == 2:
#                 ref_pt[1] = [x, y]

# count = 0
# while True:
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align.process(frames)

#     aligned_color_frame = aligned_frames.get_color_frame()
#     aligned_depth_frame = aligned_frames.get_depth_frame()

#     if not aligned_color_frame and not aligned_depth_frame:
#         continue

#     aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
#     aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

#     cv2.namedWindow("image")
#     cv2.setMouseCallback("image", click_and_crop)

#     tmp_frame = aligned_color_image[:,:,::-1].copy()
#     if len(ref_pt) == 2:
#         cv2.rectangle(tmp_frame, ref_pt[0], ref_pt[1], (0,255,0))
    
#     cv2.imshow('image', tmp_frame)   
#     key = cv2.waitKey(1)

#     if (len(ref_pt) == 3):
#         ref_pt[1] = ref_pt[2]
#         break
#     elif key == ord('r'):
#         is_selecting = False
#         ref_pt.clear()

# cv2.destroyAllWindows()

# # 8: beaker, 9: flask
# obj_type = 9
# bbox = np.array((ref_pt[0], ref_pt[1]), dtype=np.float32).tobytes()
# rgbd = np.dstack((aligned_color_image, aligned_depth_image)).astype(np.float32).tobytes()

# print(np.dstack((aligned_color_image, aligned_depth_image)).astype(np.float32).shape)

# socket.send(bytes([int(obj_type)]) + bbox + rgbd)
# pose_matrix = np.frombuffer(socket.recv(), dtype=np.float32).reshape(4, 4)

# predefined_pose = np.eye(4,4)
# if obj_type == 9:
#     predefined_pose[0, 3] += 0.13
#     predefined_pose[:3, :3] = R.from_euler('xyz', [0, 180, -45], degrees=True).as_matrix()
# elif obj_type == 8:
#     predefined_pose[0, 3] += 0.038
#     predefined_pose[2, 3] += 0.095
#     predefined_pose[:3, :3] = R.from_euler('xyz', [0, 180, 45], degrees=True).as_matrix()

# cam2target = pose_matrix @ predefined_pose

# print("==================================================")
# print('cam2target:', cam2target)

# # tvecs1 = pose_matrix[:3,3].reshape(1,3)
# # rvecs1 = R.from_matrix(pose_matrix[:3,:3]).as_rotvec().reshape(1,3)

# # bgr = aligned_color_image[:,:,::-1].copy()
# # cv2.drawFrameAxes(bgr, K, None, rvecs1, tvecs1, 0.05)
# # cv2.imshow('Pose Result', bgr)

# bgr = aligned_color_image[:,:,::-1].copy()
# tvecs = cam2target[:3,3].reshape(1,3)
# rvecs = R.from_matrix(cam2target[:3,:3]).as_rotvec().reshape(1,3)

# cv2.drawFrameAxes(bgr, K, None, rvecs, tvecs, 0.05)
# cv2.imshow('Pose Result', bgr)
# key = cv2.waitKey()
# if key != ord('s'):
#     exit(0)
# cv2.destroyAllWindows()

# flag = bytes([int(NetProto.REQ_Q)])
# socket_control.send(flag)
# current_q = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

# base2ee = np.array(robot.fkine(current_q))
# base2target = base2ee @ ee2cam @ cam2target

# base2target[0, 3] -= 0.01
# base2target[2, 3] += 0.02

# target2base = np.linalg.inv(base2target)
# target2base[2, 3] += 0.1
# base2target_pre = np.linalg.inv(target2base)

# base2target_after = base2target.copy()
# base2target_after[2, 3] += 0.07

# Tep = SE3.CopyFrom(base2target_pre, check=False)
# sol = robot.ik_LM(Tep, q0=current_q)
# pred_matrix = robot.fkine(sol[0])
# distance = np.linalg.norm(base2target_pre[:3,3] - np.array(pred_matrix)[:3,3])
# if distance > 0.05:
#     print("Couldn't solve the IK1")
#     exit(0)
# q1 = np.array(sol[0], dtype=np.float64)

# Tep2 = SE3.CopyFrom(base2target, check=False)
# sol2 = robot.ik_LM(Tep2, q0=q1)
# pred_matrix2 = robot.fkine(sol2[0])
# distance = np.linalg.norm(base2target[:3,3] - np.array(pred_matrix2)[:3,3])
# if distance > 0.05:
#     print("Couldn't solve the IK2")
# q2 = np.array(sol2[0], dtype=np.float64)

# Tep3 = SE3.CopyFrom(base2target_after, check=False)
# sol3 = robot.ik_LM(Tep3, q0=q2)
# pred_matrix3 = robot.fkine(sol3[0])
# distance = np.linalg.norm(base2target_after[:3,3] - np.array(pred_matrix3)[:3,3])
# if distance > 0.05:
#     print("Couldn't solve the IK3")
# q3 = np.array(sol3[0], dtype=np.float64)

# filename = f'./tmp/{models[obj_type-1]}.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 30.0
# frame_size = (640, 480)

# out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

# flag = bytes([int(NetProto.MOVE_ARM_BY_Q)])  # q1 으로 움직임
# socket_control.send(flag+q1.tobytes())
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
#         out.write(img)
#         cv2.imshow('img', img)
#         key = cv2.waitKey(1)

# flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q2로 움직임
# socket_control.send(flag+q2.tobytes())
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
#         out.write(img)
#         cv2.imshow('img', img)
#         key = cv2.waitKey(1)

# flag = bytes([int(NetProto.GRIPPER_CLOSE)]) # 그리퍼 움직임
# socket_control.send(flag)
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
#         out.write(img)
#         cv2.imshow('img', img)
#         key = cv2.waitKey(1)

# flag = bytes([int(NetProto.MOVE_ARM_BY_Q)]) # q3로 움직임
# socket_control.send(flag+q3.tobytes())
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
#         out.write(img)
#         cv2.imshow('img', img)
#         key = cv2.waitKey(1)

# cv2.destroyAllWindows()


if __name__ == '__main__':
    data_dir = "/home/vision/packages/FoundationPose/result"
    scene = data_load_image(data_dir)



    object_model = data_load_cad(data_dir)

    # data_dir = "/home/vision/packages/Scale-Balanced-Grasp"
    data_dir = "/home/vision/packages/gamma/graspness_implementation"

    grasp_info = data_load_grasp(data_dir)

    calib_info = data_load_calib(ROOT_DIR)
    

    gripper = create_gripper('panda')
    gripper_o3d = gripper.to_open3d()
    print(np.array(grasp_info['transformation']))
    gripper_o3d.transform(np.array(grasp_info['transformation']))

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    
    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    gripper_trans = np.array(grasp_info['transformation'])
    half_depth_vector = np.array([0, 0, 0.109])
    gripper_trans[:3, 3] = gripper_trans[:3, 3] + gripper_trans[:3, :3] @ half_depth_vector
    


    gripper_frame.transform(gripper_trans)

    robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])  # 크기 0.1, 원점 [0, 0, 0]
    b2c = np.array(calib_info['c2b'])
    b2c[:3, 3] /= 1000.0
    
    robot_base_frame.transform(b2c)
    print(np.array(b2c))

    


    o3d.visualization.draw_geometries([scene,object_model,gripper_o3d,camera_frame,gripper_frame])

