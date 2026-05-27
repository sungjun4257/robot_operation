from sympy import *
import math
import numpy as np

class Direct_Kinematics():
    def __init__(self):
        self.joint_var = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def dh_params(self):
        M_PI = math.pi

        # Create DH parameters (data given by maker franka-emika)
        # self.dh = [ [0,         0,        0.333,   self.joint_var[0]],
        #             [-M_PI/2,   0,        0,       self.joint_var[1]],
        #             [ M_PI/2,   0,        0.316,   self.joint_var[2]],
        #             [ M_PI/2,   0.0825,   0,       self.joint_var[3]],
        #             [-M_PI/2,  -0.0825,   0.384,   self.joint_var[4]],
        #             [ M_PI/2,   0,        0,       self.joint_var[5]],
        #             [ M_PI/2,   0.088,    0.107 + 0.1034,   self.joint_var[6]]]
        
        self.dh =  [[ 0.        ,  0.        ,  0.333     ,  0.         + self.joint_var[0]],
                    [-1.57302966, -0.00483488, -0.0001403 ,  0.00068802 + self.joint_var[1]],
                    [ 1.57298024,  0.00488819,  0.31678138,  0.00476678 + self.joint_var[2]],
                    [ 1.56766203,  0.08387579, -0.00232793, -0.0078252  + self.joint_var[3]],
                    [-1.5696951 , -0.08816632,  0.3862611 ,  0.02148609 + self.joint_var[4]],
                    [ 1.56556072,  0.00314022, -0.00342299,  0.00009337 + self.joint_var[5]],
                    [ 1.56791435,  0.08569503,  0.107 + 0.1034 ,  0.    + self.joint_var[6]],]
        
        return self.dh
      
    def TF_matrix(self,i,dh):
        # Define Transformation matrix based on DH params
        alpha = dh[i][0]
        a = dh[i][1]
        d = dh[i][2]
        q = dh[i][3]
        
        TF = Matrix([[cos(q),-sin(q), 0, a],
                    [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
                    [sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
                    [   0,  0,  0,  1]])
        return TF

    def get_transform(self, q):
        self.joint_var = []
        for i in range(0,7):
            self.joint_var.append(q[i])

        dh_parameters = self.dh_params()

        T_01 = self.TF_matrix(0,dh_parameters)
        T_12 = self.TF_matrix(1,dh_parameters)
        T_23 = self.TF_matrix(2,dh_parameters)
        T_34 = self.TF_matrix(3,dh_parameters)
        T_45 = self.TF_matrix(4,dh_parameters)
        T_56 = self.TF_matrix(5,dh_parameters)
        T_67 = self.TF_matrix(6,dh_parameters)

        T_07 = T_01*T_12*T_23*T_34*T_45*T_56*T_67

        return np.array(T_07).astype(np.float32)