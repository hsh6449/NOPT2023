import numpy as np
import math
from utils import *

def linear_conjugate(f, x0, max_iter=10000, eps=1e-5, verbose=False):
    x = x0[0]
    y = x0[1]

    rx, ry = gradient(f, x, y) # r
    dx = -rx # direction, p
    dy = -ry

    A = np.array([[1,0],[0,1]])

    while (rx == 0) & (ry == 0) :

      for i in range(max_iter):
          if verbose:
              print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f(x,y)}")
          if norm(g) < eps:
              break

          alpha = inexact_line_search(f, x, y, dx, dy)

          x = x + alpha * d
          y = y + alpha * d

          rx = rx + alpha * A * dx # A를 어떻게 정의할 것인지
          ry = ry + alpha * A * dy
          
          beta_x = (rx.T @ rx) / (rx.T @ rx)
          dx = -rx + beta_x * dx
          dy = -ry + beta_y * dy

    return x, y

def cg_fr(f, x0, max_iter=1000, eps=1e-5, verbose=False):
    x = x0[0]
    y = x0[1]

    fxy = f(x,y)
    rx, ry = gradient(f, x, y) # r

    dx = -rx # direction, p
    dy = -ry

    while (rx == 0) and (ry == 0) :

      for i in range(max_iter):
          if verbose:
              print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f(x,y)}")
          if norm(g) < eps:
              break

          alpha = inexact_line_search(f, x, y, dx, dy)

          x = x + alpha * dx
          y = y + alpha * dy

          prev_gx = rx
          prev_gy = ry

          rx, ry = gradient(f, x, y)

          beta_x = (new_gx.T @ new_gx) / (prev_gx.T @ prev_gy)
          beta_y = (new_gy.T @ new_gy) / (prev_gy.T @ prev_gy)

          dx = -new_gx + beta_x * dx
          dy = -new_gy + beta_y * dy

    return x, y 

def cg_pr(f, x0, max_iter=1000, eps=1e-5, verbose=False):
    x = x0[0]
    y = x0[1]

    rx, ry = gradient(f, x, y) # r

    dx = -rx # direction, p
    dy = -ry

    while (rx == 0) and (ry == 0) :

      for i in range(max_iter):
          if verbose:
              print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f(x,y)}")
          if norm(g) < eps:
              break

          alpha = inexact_line_search(f, x, y, dx, dy)

          x = x + alpha * dx
          y = y + alpha * dy

          prev_gx = rx
          prev_gy = ry

          rx, ry = gradient(f, x, y)

          beta_x = (new_gx.T @ (new_gx - prev_gx)) / (prev_gx.T @ prev_gy)
          beta_y = (new_gy.T @ (new_gy - prev_gy)) / (prev_gy.T @ prev_gy)

          dx = -new_gx + beta_x * dx
          dy = -new_gy + beta_y * dy

    return x, y 

def cg_hs(f, x0, max_iter=1000, eps=1e-5, verbose=False):
    x = x0[0]
    y = x0[1]

    rx, ry = gradient(f, x, y) # r

    dx = -rx # direction, p
    dy = -ry

    while (rx == 0) and (ry == 0) :

      for i in range(max_iter):
          if verbose:
              print(f"[iter] : {i}, -- [x,y] : {x,y}, -- [f(x,y)] : {f(x,y)}")
          if norm(g) < eps:
              break

          alpha = inexact_line_search(f, x, y, dx, dy)

          x = x + alpha * dx
          y = y + alpha * dy

          prev_gx = rx
          prev_gy = ry

          rx, ry = gradient(f, x, y)

          beta_x = (new_gx.T @ (new_gx.T - prev_gx)) / ((new_gx - prev_gx).T @ prev_gx)
          beta_y = (new_gy.T @ (new_gy.T - prev_gy)) / ((new_gy - prev_gy).T @ prev_gy)

          dx = -new_gx + beta_x * dx
          dy = -new_gy + beta_y * dy

    return x, y