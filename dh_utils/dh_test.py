from dh_param import Direct_Kinematics
from DH_Panda import Panda

import numpy as np
np.set_printoptions(suppress=True)
import math

import yaml

from scipy.spatial.transform import Rotation as R

dh = Direct_Kinematics()
q = np.random.randint(1, 100, size=7)
q = q / q.max()

M_PI = math.pi

param = np.array(np.load('./dh_param.npy'))
robot = Panda(param)

print(dh.get_transform(q).astype(np.float16))
print(np.array(robot.fkine(q)).astype(np.float16))

b2e = np.array(robot.fkine(q))

with open('./hand_eye_v1_mean_excl_0123_27.yaml', 'r') as f:
    data = yaml.load(f, yaml.FullLoader)
    e2c = np.array(data['e2c'])
    e2c[:3, 3] /= 1000

rot_45 = np.eye(4, 4)
rot_45[:3, :3] = R.from_euler('xyz', angles=[0, 0, 45], degrees=True).as_matrix()

c2t = np.array([[ 0.51243246,  0.14999227,  0.84552652,  0.04220741],
        [-0.74468201, -0.41269931,  0.52452642, -0.00624416],
        [ 0.42762306, -0.89843297, -0.09978402,  0.68418735],
        [ 0.        ,  0.        ,  0.        ,  1.        ],])

print(b2e @ rot_45 @ e2c @ c2t)
print(R.from_matrix(e2c[:3, :3]).as_euler('xyz', degrees=True))

e2c = rot_45 @ e2c
np.save('ee2cam.npy', e2c)

print(b2e @ e2c @ c2t)
print(R.from_matrix(e2c[:3, :3]).as_euler('xyz', degrees=True))