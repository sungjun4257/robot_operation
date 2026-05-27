import numpy as np
import cv2

import zmq
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

from dh_utils.DH_Panda import Panda
from spatialmath import SE3

from enum import IntEnum
import time

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

param = np.array(np.load('./dh_utils/param/dh_param_1024.npy'))
robot = Panda(param)

ee2cam = np.load('./dh_utils/param/ee2cam_1024.npy')

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

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:1111")

context_control = zmq.Context()
socket_control = context_control.socket(zmq.REQ)
socket_control.connect("tcp://192.168.100.150:5555")

is_selecting = False
ref_pt = []
def click_and_crop(event, x, y, flags, param):
    global is_selecting
    
    if event == cv2.EVENT_LBUTTONDOWN:
        is_selecting = True
        ref_pt.append([x, y])
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if len(ref_pt) == 2:
            is_selecting = False
            ref_pt.append([x, y])
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_selecting == True:
            if len(ref_pt) == 1:
                ref_pt.append([x, y])
            elif len(ref_pt) == 2:
                ref_pt[1] = [x, y]

count = 0
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_color_frame = aligned_frames.get_color_frame()
    aligned_depth_frame = aligned_frames.get_depth_frame()

    if not aligned_color_frame and not aligned_depth_frame:
        continue

    aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    tmp_frame = aligned_color_image[:,:,::-1].copy()
    if len(ref_pt) == 2:
        cv2.rectangle(tmp_frame, ref_pt[0], ref_pt[1], (0,255,0))
    
    cv2.imshow('image', tmp_frame)   
    key = cv2.waitKey(1)

    if (len(ref_pt) == 3):
        ref_pt[1] = ref_pt[2]
        break
    elif key == ord('r'):
        is_selecting = False
        ref_pt.clear()

cv2.destroyAllWindows()

# 8: beaker, 9: flask
obj_type = 9
bbox = np.array((ref_pt[0], ref_pt[1]), dtype=np.float32).tobytes()
rgbd = np.dstack((aligned_color_image, aligned_depth_image)).astype(np.float32).tobytes()

print(np.dstack((aligned_color_image, aligned_depth_image)).astype(np.float32).shape)

socket.send(bytes([int(obj_type)]) + bbox + rgbd)
pose_matrix = np.frombuffer(socket.recv(), dtype=np.float32).reshape(4, 4)

predefined_pose = np.eye(4,4)
if obj_type == 9:
    predefined_pose[0, 3] += 0.13
    predefined_pose[:3, :3] = R.from_euler('xyz', [0, 180, -45], degrees=True).as_matrix()
elif obj_type == 8:
    predefined_pose[0, 3] += 0.038
    predefined_pose[2, 3] += 0.095
    predefined_pose[:3, :3] = R.from_euler('xyz', [0, 180, 45], degrees=True).as_matrix()

cam2target = pose_matrix @ predefined_pose

print("==================================================")
print('cam2target:', cam2target)

# tvecs1 = pose_matrix[:3,3].reshape(1,3)
# rvecs1 = R.from_matrix(pose_matrix[:3,:3]).as_rotvec().reshape(1,3)

# bgr = aligned_color_image[:,:,::-1].copy()
# cv2.drawFrameAxes(bgr, K, None, rvecs1, tvecs1, 0.05)
# cv2.imshow('Pose Result', bgr)

bgr = aligned_color_image[:,:,::-1].copy()
tvecs = cam2target[:3,3].reshape(1,3)
rvecs = R.from_matrix(cam2target[:3,:3]).as_rotvec().reshape(1,3)

cv2.drawFrameAxes(bgr, K, None, rvecs, tvecs, 0.05)
cv2.imshow('Pose Result', bgr)
key = cv2.waitKey()
if key != ord('s'):
    exit(0)
cv2.destroyAllWindows()

flag = bytes([int(NetProto.REQ_Q)])
socket_control.send(flag)
current_q = np.frombuffer(socket_control.recv()[1:], dtype=np.float64)

base2ee = np.array(robot.fkine(current_q))
base2target = base2ee @ ee2cam @ cam2target

base2target[0, 3] -= 0.01
base2target[2, 3] += 0.02

target2base = np.linalg.inv(base2target)
target2base[2, 3] += 0.1
base2target_pre = np.linalg.inv(target2base)

base2target_after = base2target.copy()
base2target_after[2, 3] += 0.07

Tep = SE3.CopyFrom(base2target_pre, check=False)
sol = robot.ik_LM(Tep, q0=current_q)
pred_matrix = robot.fkine(sol[0])
distance = np.linalg.norm(base2target_pre[:3,3] - np.array(pred_matrix)[:3,3])
if distance > 0.05:
    print("Couldn't solve the IK1")
    exit(0)
q1 = np.array(sol[0], dtype=np.float64)

Tep2 = SE3.CopyFrom(base2target, check=False)
sol2 = robot.ik_LM(Tep2, q0=q1)
pred_matrix2 = robot.fkine(sol2[0])
distance = np.linalg.norm(base2target[:3,3] - np.array(pred_matrix2)[:3,3])
if distance > 0.05:
    print("Couldn't solve the IK2")
q2 = np.array(sol2[0], dtype=np.float64)

Tep3 = SE3.CopyFrom(base2target_after, check=False)
sol3 = robot.ik_LM(Tep3, q0=q2)
pred_matrix3 = robot.fkine(sol3[0])
distance = np.linalg.norm(base2target_after[:3,3] - np.array(pred_matrix3)[:3,3])
if distance > 0.05:
    print("Couldn't solve the IK3")
q3 = np.array(sol3[0], dtype=np.float64)

filename = f'./tmp/{models[obj_type-1]}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
frame_size = (640, 480)

out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

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
        out.write(img)
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
        out.write(img)
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
        out.write(img)
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
        out.write(img)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)

cv2.destroyAllWindows()
