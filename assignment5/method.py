import numpy as np
import math
from utils import *
import time

def linear_conjugate(f, x0, max_iter=100, eps=1e-8, verbose=False):

    print("Using Method : linear_conjugate\n")
    x = x0[0]
    y = x0[1]

    r = gradient(f, x, y) # r
    dx = -r[0] # direction, p
    dy = -r[1]

    iteration = 0

    while (r[0] != 0) & (r[1] != 0) :

        iteration += 1
        print(r)

        if verbose:
            print(f"[iter] : {iteration}, -- [x,y] : {x,y}, -- [f(x,y)] : {f([x,y])}")
        if (np.linalg.norm(r[0]) < eps) & (np.linalg.norm(r[1]) < eps) :
            print(np.linalg.norm(rx))
            print(np.linalg.norm(ry))
            return x, y
        
        if iteration == max_iter -1 :
            return x, y

        alpha = inexact_line_search_w(f, x, y, dx, dy)

        x, y = x + alpha[0] * dx, y + alpha[1] * dy

        rx, ry = gradient(f, x, y)
        
        r = np.array([rx, ry])
          
        beta = (r.T @ r) / ((r.T @ r) + 1e-5)
        
        dx = -r[0] + beta * dx
        dy = -r[1] + beta * dy        

    return x, y

def cg_fr(f, x0, max_iter=10000, eps=1e-5, verbose=False):
    print("Using Method : cg_fr\n")
    x = x0[0]
    y = x0[1]

    fxy = f(x0)
    rx, ry = gradient(f, x, y) # r

    dx = -rx # direction, p
    dy = -ry
    d = np.array([dx, dy])

    print(rx, ry)
    
    start = time.time()
    while (rx != 0) and (ry != 0) :

      for i in range(max_iter):
        if verbose:
            print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f([x,y])}")
        if (np.linalg.norm(rx) < eps) & (np.linalg.norm(ry) < eps):
            end = time.time()
            return x, y
        
        if i == max_iter -1 :
            end = time.time()
            return x, y

        alpha = inexact_line_search_w(f, x, y, dx, dy)

        x = x + alpha[0] * d[0]
        y = y + alpha[1] * d[1]

        prev_gx = rx
        prev_gy = ry

        prev_g = np.array([prev_gx, prev_gy])

        rx, ry = gradient(f, x, y)
        r = np.array([rx, ry])

        beta_x = (rx.T * rx) / (prev_gx.T * prev_gx)
        beta_y = (ry.T * ry) / (prev_gy.T * prev_gy) #각각 나누어 구하기
        beta = (r.T @ r) / (prev_g.T @ prev_g) # 행렬연산으로 한번에 구하기 

        dx = -rx + beta_x * dx
        dy = -rx + beta_y * dy
        d = -r + beta * np.array([dx, dy])

        prev_f = fxy
        fxy = f([x,y])

        if (prev_f < fxy) & (abs(fxy - prev_f) > 1):
            end = time.time()
            print("=========================Optimization Complete!=========================")
            print(f"[Terminal Result] - [iter] : {i}, [x] : {x}, [y] : {y}, f(x,y) : {fxy}, [Time] : {end - start}\n")
            return x, y

    return x, y 

def cg_pr(f, x0, max_iter=10000, eps=1e-5, verbose=False):
    print("Using Method : cg_pr\n")
    x = x0[0]
    y = x0[1]

    fxy = f(x0)
    rx, ry = gradient(f, x, y) # r

    dx = -rx # direction, p
    dy = -ry

    print(rx, ry)

    start = time.time()

    while (rx != 0) and (ry != 0) :

      for i in range(max_iter):
          if verbose:
              print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f([x,y])}")
          if (np.linalg.norm(rx) < eps) & (np.linalg.norm(ry) < eps):
              
            end = time.time()
            print(f"time : , {end - start}")
            return x, y
          
          if i == max_iter -1 :
            end = time.time()
            print(f"time : , {end - start}")
            return x, y

          alpha = inexact_line_search_w(f, x, y, dx, dy)

          x = x + alpha[0] * dx
          y = y + alpha[1] * dy

          prev_gx = rx
          prev_gy = ry

          rx, ry = gradient(f, x, y)

          beta_x = (rx.T * (rx - prev_gx)) / (prev_gx.T * prev_gy)
          beta_y = (ry.T * (ry - prev_gy)) / (prev_gy.T * prev_gy)

          dx = -rx + beta_x * dx
          dy = -rx + beta_y * dy

          prev_f = fxy
          fxy = f([x,y])

          if (prev_f < fxy) & (abs(fxy - prev_f) > 1):
            end = time.time()
            print("=========================Optimization Complete!=========================")
            print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {fxy}, [Time] : {end - start}\n")
            return x, y

    return x, y 


def cg_hs(f, x0, max_iter=1000, eps=1e-5, verbose=False):
    print("Using Method : cg_hs\n")
    x = x0[0]
    y = x0[1]

    fxy = f(x0)
    rx, ry = gradient(f, x, y) # r

    dx = -rx # direction, p
    dy = -ry

    print(rx, ry)

    start = time.time()
    while (rx != 0) and (ry != 0) :

      for i in range(max_iter):
        if verbose:
            print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f([x,y])}")
        if (np.linalg.norm(rx) < eps) & (np.linalg.norm(ry) < eps):
            end = time.time()
            print(f"time : , {end - start}")
            return x, y
          
        if i == max_iter -1 :
            end = time.time()
            print(f"time : , {end - start}")
            return x, y

        alpha = inexact_line_search_w(f, x, y, dx, dy)

        x = x + alpha[0] * dx
        y = y + alpha[1] * dy

        prev_gx = rx
        prev_gy = ry

        rx, ry = gradient(f, x, y)

        beta_x = (rx.T * (rx.T - prev_gx)) / ((rx - prev_gx).T * prev_gx)
        beta_y = (ry.T * (ry.T - prev_gy)) / ((ry - prev_gy).T * prev_gy)

        dx = -rx + beta_x * dx
        dy = -rx + beta_y * dy
        
        prev_f = fxy
        fxy = f([x,y])

        if (prev_f < fxy) & (abs(fxy - prev_f) > 1):
            end = time.time()
            print("=========================Optimization Complete!=========================")
            print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {fxy}, [Time] : {end - start}\n")
            return x, y 

    return x, y 

