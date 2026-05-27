import numpy as np
np.set_printoptions(suppress=True, precision=6)

from scipy.spatial.transform import Rotation as R

dh_param = np.array([[  0.2228    ,   0.0002332 ,   0.3312573 ,  -0.0829    ],
                    [-89.83648259,  -0.00206009,  -0.00068258,  -0.32552132],
                    [ 89.78734358,   0.00087231,   0.3146078 ,   0.28575524],
                    [ 90.04798433,   0.08542625,   0.0002838 ,  -0.27221488],
                    [-90.0320432 ,  -0.08383469,   0.38798413,   1.33016332],
                    [ 89.71623825,   0.0022131 ,   0.00230517,   0.21921158],
                    [ 90.04554927,   0.07890732,   0.0001124 ,  -0.5208    ]])

dh_param[:, 0] *= (np.pi / 180)
dh_param[:, 3] *= (np.pi / 180)

print(dh_param)

e2c = np.array([[0.9987  , -0.05078 , -0.009633, -31.79],     
                [0.05083 ,  0.9987  ,  0.005193,  51.13] ,    
                [0.009357, -0.005675,  0.9999  , -36.84] ,    
                [0       ,  0       ,  0       ,  1    ]])

e2c[:3, 3] /= 1000

rot_z = np.eye(4,4)
rot_z[:3,:3] = R.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix()
e2c = rot_z @ e2c

print(e2c)
print(R.from_matrix(e2c[:3,:3]).as_euler('xyz',degrees=True))

# np.save('./param/dh_param_ketti1.npy', dh_param)
# np.save('./param/ee2cam_ketti1.npy', e2c)

# dh_pre = np.load('./param/dh_param_1024.npy')
# e2c_pre = np.load('./param/ee2cam_1024.npy')

# print(dh_pre)
# print(e2c_pre)
# print(R.from_matrix(e2c_pre[:3,:3]).as_euler('xyz',degrees=True))