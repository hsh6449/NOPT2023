import math
import numpy as np
import pdb
import time

from utils import *

def Levenberg_Marquardt(f, data, target_w = 1, tol=1e-6, MaxIter=20):
        
    x, y , z = data['x'], data['y'], data['z']

    if target_w == 1 :
        target = data["data1"]
    elif target_w == 2 :
        target = data["data2"]

    iter = 1

    P = np.array([1,1,1,1], dtype='float64') # initial coefficient setting
    l = 0.1 # initial lambda setting

    start = time.time()
    while iter < MaxIter:
        f_val = f(x, y, z, P)

        r = target - f_val # residual

        ## termination condition
        if np.linalg.norm(r) < tol:
            return P

        ## update P
        J_T = jaccobian(f, x, y, z, P) # jaccobian, 구현한 jacobbian matrix의 shape이 예상과 다르게 나와서 편의상 J_T로 표현
        J = np.transpose(J_T) # transpose of jaccobian
        J_T_J = np.matmul(J_T, J) # J_T * J

        J_T_J_inv = np.linalg.pinv(J_T_J + l*np.diag(J_T_J)) # (J_T * J + lambda * I)^-1

        P_new = P + J_T_J_inv.dot(J_T).dot(r) # 임시 P 
        J_T_J_new = np.matmul(np.transpose(jaccobian(f, x, y, z, P_new)), jaccobian(f, x, y, z, P_new)) # 임시 P로 구한 J_T * J

        ## update lambda
        """
        determinant로 행렬크기를 비교하여 lambda를 업데이트하는 방법을 사용하였다.
        """
        if np.linalg.det(J_T_J) < np.linalg.det(J_T_J_new):
            l = l*10
            break
        else:
            l = l/10
            P = P_new

        prev_f = f_val
        f_val = f(x, y, z, P)

        r = target - f_val
        rmse = np.sqrt(np.sum(r**2)/len(r))

        print("iter : ", iter, "rmse : ", rmse, "time : ", time.time() - start)

        iter += 1
    print("final P : ", P)
    return P



def Gauss_Newton(f, data, target_w = 1, tol=1e-11, MaxIter=30):

    x, y , z = data['x'], data['y'], data['z']

    if target_w == 1 :
        target = data["data1"]
    elif target_w == 2 :
        target = data["data2"]

    iter = 1

    P = np.array([1,1,1,1], dtype='float64') # initial coefficient setting

    start = time.time()
    while iter < MaxIter:
        f_val = f(x, y, z, P) # f value

        r = target - f_val # residual

        ## termination condition
        if np.linalg.norm(r) < tol:
            return P

        ## update P
        J_T = jaccobian(f, x, y, z, P)
        J = np.transpose(J_T)
        J_T_J = np.matmul(J_T, J)

        J_T_J_inv = np.linalg.pinv(J_T_J)

        P = P + J_T_J_inv.dot(J_T).dot(r)

        f_val = f(x, y, z, P)

        ## caculate rmse
        r = target - f_val
        rmse = np.sqrt(np.sum(r**2)/len(r))

        print("iter : ", iter, "rmse : ", rmse, "time : ", time.time() - start)


        iter += 1
    print("final P : ", P)
    return P

