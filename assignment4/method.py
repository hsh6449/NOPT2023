import numpy as np
# import math
from utils import *
import time

def steepest_descent(f, init_x=1.2, init_y=1.2, epsilon=1e-4, max_iter=10000):

  iter = 1
  
  x = init_x
  y = init_y

  coordinates = []
  results = []

  start = time.time()
  print("-------------------------Steepest Descent Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f([x,y])} \n")

  prev_f = f([x,y])

  while iter <= max_iter:

    coordinates.append((x,y))
    results.append(prev_f)

    dx, dy = gradient(f, x, y) # derivative of f(x,y)

    # alpha = bisection(pi, 0, 2)
    alpha = inexact_line_search(f, x, y, dx, dy)

    x = x - alpha[0] * dx
    y = y - alpha[1] * dy

    print(f"[optimizing...] - [iter] : {iter}, [alpha] : {alpha}, [Direction] : {(dx,dy)}, [x] : {x}, [y] : {y}, [f(x,y)] : {f([x,y])}")

    
    if abs(f([x,y]) - prev_f) < epsilon:
      end = time.time()
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}, [Time] : {end - start}\n")
      return coordinates, results
    
    
    prev_f = f([x,y])
    iter += 1
  
  end = time.time()
  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}, [Time] : {end - start}\n")

  return coordinates, results

def newton(f, init_x = 1.2 , init_y = 1.2, epsilon=0.0001 , max_iter = 1000):

  x = init_x
  y = init_y

  x_value = [x, y]

  coordinates = []
  results = []

  start = time.time()
  print("-------------------------Newton Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f(x_value)} \n")

  iter = 1

  prev_f = f(x_value)

  while iter <= max_iter:

    coordinates.append((x,y))
    results.append(prev_f)
    
    hessian = hessian_matrix(f, x, y)
    # print(hessian)
    inv_hessian = np.linalg.inv(hessian)
    
    grad = gradient(f, x, y)

    x = x - inv_hessian[0][0] * grad[0] - inv_hessian[0][1] * grad[1]
    y = y - inv_hessian[1][0] * grad[0] - inv_hessian[1][1] * grad[1]

    x_value = np.array([x, y])
    # x_value = x_value - np.dot(inv_hessian, np.array(grad))


    print(f"[optimizing...] - [iter] : {iter}, [Direction] : {grad}, [x] : {x}, [y] : {y}, [f(x,y)] : {f(x_value)}")
    
    if abs(f(x_value) - prev_f) < epsilon:
      end = time.time()
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
      return coordinates, results
    if abs(f(x_value) - prev_f) < epsilon : # 수렴하는 경우 
      end = time.time()
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
      return coordinates, results
    
    iter += 1
    prev_f = f(x_value)

  end = time.time()
  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
  return coordinates, results



def SR1(f, init_x = 1.2 , init_y = 1.2, epsilon=0.000001 , max_iter = 1000):

  # prev_f = f(x_value)
    
  x = init_x
  y = init_y

  x_value = np.array([x, y])

  start = time.time()
  print("-------------------------SR1 Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f(x_value)}\n")

  iter = 1

  coordinates = []
  results = []

  B = np.array([[1.0,0.0],[0.0,1.0]])
  inv_B = np.linalg.inv(B)

  prev_f = f(x_value)

  while iter <= max_iter:
      
      coordinates.append((x,y))
      results.append(prev_f)

      grad = gradient(f, x_value[0], x_value[1])
      alpha = inexact_line_search(f, x_value[0], x_value[1], grad[0], grad[1])
      
      s = np.array([alpha[0] * grad[0], alpha[1] * grad[1]])
      grad_new = gradient(f, x_value[0] + alpha[0] * grad[0], x_value[1] + alpha[1] * grad[1])

      yk = grad_new - grad

      # B += np.outer(yk - np.dot(B, s), yk - np.dot(B, s)) / (np.dot(yk - np.dot(B, s), s) + 1e-2)

      # inv_B = np.linalg.inv(B)
      
      # if (yk - np.dot(B, s)).T.dot(s) >= 0.2 * np.linalg.norm(s) * np.linalg.norm(yk - np.dot(B, s)) >0 :
          # B = B + (yk - np.dot(B, s)).dot((yk - np.dot(B, s)).T) / ((yk - np.dot(B,s)).T.dot(s) +1e-5)
          # inv_B = inv_B + (s - inv_B.dot(yk)).dot((s - inv_B.dot(yk)).T) / ((s - inv_B.dot(yk)).T.dot(yk) +1e-5)
      # else:
          # B = B 
          # inv_B = inv_B
      inv_B = inv_B + (s - inv_B.dot(yk)).dot((s - inv_B.dot(yk)).T) / ((s - inv_B.dot(yk)).T.dot(yk) +1e-6)
      # inv_B = np.linalg.inv(B)
      
      x_value =  x_value - np.dot(inv_B, np.array(grad)).dot(alpha)
      # x = x_value[0] - inv_B[0][0]*grad[0] * alpha[0] - inv_B[0][1]*grad[1] * alpha[1]
      # y = x_value[1] - inv_B[1][0]*grad[0] * alpha[0] - inv_B[1][1]*grad[1] * alpha[1]

      # x_value = np.array([x, y])

      print(f"[optimizing...] - [iter] : {iter}, [Alpha] : {alpha}, [Direction] : {B}, [x] : {x_value[0]}, [y] : {x_value[1]}, [f(x,y)] : {f(x_value)}")

      
      if (prev_f < f(x_value)) & (abs(f(x_value) - prev_f) < 1):
          end = time.time()
          print("=========================Optimization Complete!=========================")
          print(f"[Terminal Result] - [iter] : {iter}, [x] : {x_value[0]}, [y] : {x_value[1]}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
          return coordinates, results
      
      if abs(f(x_value) - prev_f) < epsilon : # 수렴하는 경우 
        end = time.time()
        print("=========================Optimization Complete!=========================")
        print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
        return coordinates, results
      
      prev_f = f(x_value)
      iter += 1

  end = time.time()
  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x_value[0]}, [y] : {x_value[1]}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
  return coordinates, results


def BFGS(f, init_x = 1.2 , init_y = 1.2, epsilon=0.001 , max_iter = 1000):

  x = init_x
  y = init_y

  x_value = np.array([x, y])

  start = time.time()
  print("-------------------------BFGS Method-------------------------")
  print(f"[Starting Point] - [x] : {x}, [y] : {y}, [fxy] : {f(x_value)} \n")

  iter = 1

  coordinates = []
  results = []

  B = np.array([[1.0,0.0],[0.0,1.0]])

  prev_f = f(x_value)

  while iter <= max_iter:

    coordinates.append((x,y))
    results.append(prev_f)
    
    grad = gradient(f, x_value[0], x_value[1])
    alpha = inexact_line_search(f, x_value[0], x_value[1], grad[0], grad[1]) #0.0003
      
    s = np.array([alpha[0] * grad[0], alpha[1] * grad[1]])
    grad_new = gradient(f, x_value[0] + alpha[0] * grad[0], x_value[1] + alpha[0] * grad[1])

    yk = grad_new - grad
    
    # original version of BFGS
    # B = B - (B.dot(s).dot(s.T) * B) / ((s.T.dot(B).dot(s)) + 1e-5) + (yk.dot(yk.T)) / ((yk.T.dot(s)) + 1e-5) 
    inv_B = np.linalg.inv(B)

    # inverse version of BFGS
    ro = 1.0 / (yk.T.dot(s) + 1e-5)
    inv_B = (np.identity(2) - ro*s.dot(yk.T)).dot(inv_B).dot(np.identity(2) - ro*yk.dot(s.T)) + ro*s.dot(s.T)

    x_value = x_value - np.dot(inv_B, np.array([grad[0], grad[1]])).dot(alpha)
    x, y = x_value

    print(f"[optimizing...] - [iter] : {iter}, [Alpha] : {alpha}, [Direction] : {B}, [x] : {x}, [y] : {y}, [f(x,y)] : {f(x_value)}")

    
    if (abs(f(x_value) - prev_f) < 1) & (prev_f < f(x_value)): # 발산하는 경우 방지 
      end = time.time()
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f([x,y])}, [Time] : {end - start}\n")
      return coordinates, results
    
    if abs(f(x_value) - prev_f) < epsilon : # 수렴하는 경우
      end = time.time()
      print("=========================Optimization Complete!=========================")
      print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
      return coordinates, results
    
    # if f(x_value) is not np.float64:
    #   break

    
    iter += 1
    prev_f = f(x_value)

  end = time.time()
  print("=========================Optimization Failed!=========================")
  print(f"[Terminal Result] - [iter] : {iter}, [x] : {x}, [y] : {y}, f(x,y) : {f(x_value)}, [Time] : {end - start}\n")
  return coordinates, results