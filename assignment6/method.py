import math
import numpy as np

from autograd import elementwise_grad as egrad
from autograd import jacobian

def Levenberg_Marquardt(f, data, target = 1, tol=1e-6, MaxIter=10000):
    
        x, y , z = data['x'], data['y'], data['z']
    
        if target == 1 :
            target = data["data1"]
        elif target == 2 :
            target = data["data2"]
    
        iter = 1
    
        J = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
    
        P = np.array([1,1,1,0])
    
        while iter < MaxIter:
            r = target - f(x, y, z, P[0], P[1], P[2])
    
            if np.linalg.norm(r) < tol:
                return P
    
            J = jacobian(f)(x, y, z, P)
            J_T = np.transpose(J)
            J_T_J = np.matmul(J_T, J)
            J_T_r = np.matmul(J_T, r)
    
            P = P + np.matmul(np.linalg.inv(J_T_J + 0.001 * np.identity(3)), J_T_r)
    
            iter += 1
    
        return P


def Gauss_Newton(f, data, target = 1, tol=1e-6, MaxIter=10000):

    x, y , z = data['x'], data['y'], data['z']

    if target == 1 :
        target = data["data1"]
    elif target == 2 :
        target = data["data2"]

    iter = 1

    J = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])

    P = np.array([1,1,1])

    while iter < MaxIter:
        r = target - f(x, y, z, P)

        if np.linalg.norm(r) < tol:
            return P

        J = jacobian(f)(x, y, z, P)
        J_T = np.transpose(J)
        J_T_J = np.matmul(J_T, J)
        J_T_r = np.matmul(J_T, r)

        P = P + np.matmul(np.linalg.inv(J_T_J), J_T_r)

        iter += 1

    return P

