import numpy as np
import math
from utils import *

def steepest_descent(f, init_x=1.2, init_y=1.2, epsilon=1e-11, max_iter=10000):

  iter = 1
  
  x = init_x
  y = init_y

  print("-------------------------Steepest Descent Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f([x,y])} \n")

  prev_f = f([x,y])

  while iter <= max_iter:

    dx, dy = gradient(f, x, y) # derivative of f(x,y)

    # alpha = bisection(pi, 0, 2)
    alpha = inexact_line_search(f, x, y, dx, dy)

    x = x - alpha * dx
    y = y - alpha * dy

    print(f"[optimizing...] - [iter] : {iter}, [alpha] : {alpha}, [Direction] : {(dx,dy)}, [x] : {x}, [y] : {y}, [f(x,y)] : {f([x,y])}")


    if np.linalg.norm(gradient(f, x,y)) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}\n")
      return x, y
    
    if abs(f([x,y]) - prev_f) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}\n")
      return x, y

    prev_f = f([x,y])
  
  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}\n")

  return x, y

def newton(f, init_x = 1.2 , init_y = 1.2, epsilon=0.0001 , max_iter = 100000 ):

  x = init_x
  y = init_y

  x_value = [x, y]

  print("-------------------------Newton Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f(x_value)} \n")

  iter = 1

  prev_f = f(x_value)

  while iter <= max_iter:
    
    hessian = hessian_matrix(f, x, y)
    # print(hessian)
    inv_hessian = np.linalg.inv(hessian)
    
    dx, dy = gradient(f, x, y)

    x = x - inv_hessian[0][0] * dx - inv_hessian[0][1] * dy
    y = y - inv_hessian[1][0] * dx - inv_hessian[1][1] * dy

    x_value = [x, y]

    print(f"[optimizing...] - [iter] : {iter}, [Direction] : {(dx,dy)}, [x] : {x}, [y] : {y}, [f(x,y)] : {f(x_value)}")

    if np.linalg.norm(gradient(f, x,y)) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}\n")
      return x, y
    
    if abs(f(x_value) - prev_f) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}\n")
      return x, y
    
    iter += 1
    prev_f = f(x_value)

  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}\n")
  return x, y



def SR1(f, init_x = 1.2 , init_y = 1.2, epsilon=0.0001 , max_iter = 100000):

  x = init_x
  y = init_y

  x_value = np.array([x, y])

  print("-------------------------SR1 Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f(x_value)} \n")

  iter = 1

  B = np.array([[1,0],[0,1]])

  prev_f = f(x_value)

  while iter <= max_iter:
    
    dx, dy = gradient(f, x_value[0], x_value[1])
    alpha = inexact_line_search(f, x_value[0], x_value[1], dx, dy)
    dx_f, dy_f = gradient(f, x_value[0] + alpha * dx, x_value[1] + alpha * dy)

    s = np.array([alpha * dx, alpha * dy])
    yk = [dx_f - dx, dy_f - dy]

    B = B + (yk - B.dot(s)).dot((yk - B.dot(s)).T) / ((yk - B.dot(s)).T).dot(s)
    inv_B = np.linalg.inv(B)

    x_value = x_value - np.dot(inv_B, np.array([dx, dy])) * alpha
    x, y = x_value

    print(f"[optimizing...] - [iter] : {iter}, [Direction] : {B}, [x] : {x}, [y] : {y}, [f(x,y)] : {f(x_value)}")

    if np.linalg.norm(gradient(f, x,y)) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}\n")
      return x, y
    
    if abs(f(x_value) - prev_f) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}\n")
      return x, y
    
    iter += 1
    prev_f = f(x_value)

  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}\n")
  return x, y


def BFGS(f, init_x = 1.2 , init_y = 1.2, epsilon=0.0001 , max_iter = 100000):

  x = init_x
  y = init_y

  x_value = np.array([x, y])

  print("-------------------------BFGS Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f(x_value)} \n")

  iter = 1

  B = np.array([[1,0],[0,1]])

  prev_f = f(x_value)

  while iter <= max_iter:
    
    dx, dy = gradient(f, x_value[0], x_value[1])
    alpha = inexact_line_search(f, x_value[0], x_value[1], dx, dy)
    dx_f, dy_f = gradient(f, x_value[0] + alpha * dx, x_value[1] + alpha * dy)

    s = np.array([alpha * dx, alpha * dy])
    yk = [dx_f - dx, dy_f - dy]

    B = B - (B.dot(s).dot(s.T).dot(B)) / (s.T.dot(B).dot(s)) + (yk.dot(yk.T)) / (yk.T.dot(s))
    
    inv_B = np.linalg.inv(B)

    x_value = x_value - np.dot(inv_B, np.array([dx, dy])) * alpha
    x, y = x_value

    print(f"[optimizing...] - [iter] : {iter}, [Direction] : {B}, [x] : {x}, [y] : {y}, [f(x,y)] : {f(x_value)}")

    if np.linalg.norm(gradient(f, x,y)) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}\n")
      return x, y
    
    if abs(f(x_value) - prev_f) < epsilon:
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}\n")
      return x, y
    
    iter += 1
    prev_f = f(x_value)

  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}\n")
  return x, y