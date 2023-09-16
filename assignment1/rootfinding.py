import numpy as np
import math
import time

MAXITER = 100
ITER = 0

def bisection(f, a, b, tol=1e-6):

  global ITER
  global MAXITER


  if ITER == 0 :
    print("-------------------------Bisection Method-------------------------")
    print(f"[Starting Point] - [a] : {a}, [b] : {b} \n")
  
  start = time.time()

  if ITER == MAXITER:
      return (a+b)/2, a, b
  
  fx = f((a+b)/2)

  if (fx == 0) or ((b-a) < tol):
    ITER += 1
    end = time.time()

    print("=========================Root Found!=========================")
    print(f"[Terminal Result] - [iter] : {ITER}, [a] : {a}, [b] : {b}, [time] : {end-start:.6f} sec")

    return (a+b)/2, a, b
  
  elif fx*f(a) < 0:
    b = (a+b)/2
    ITER += 1
    print(f"[Root Finding...] - [iter] : {ITER}, [a] : {a}, [b] : {b} ")
    return bisection(f, a, b, tol)
  
  else:
    a = (a+b)/2
    ITER += 1
    print(f"[Root Finding...] - [iter] : {ITER}, [a] : {a}, [b] : {b} ")
    return bisection(f, a, b, tol)


def newton(f, x0, tol=1e-5):

  global ITER
  global MAXITER

  if ITER is not 0 :
    ITER = 0

  h = 1e-9
  old_x = x0

  print("-------------------------Newton Method-------------------------")
  print(f"[Starting Point] - [X] : {old_x} \n")
  start = time.time()

  while ITER < MAXITER:

    fx = f(old_x)
    dfx = (f(old_x + h) - f(old_x)) / h

    new_x = old_x - fx/dfx

    ITER += 1
    print(f"[Root Finding...] - [iter] : {ITER}, [X] : {new_x} ")

    if (abs(new_x - old_x)) < tol:
      break

    old_x = new_x

  end = time.time()
  print("=========================Root Found!=========================")
  print(f"[Terminal Result] - [iter] : {ITER}, [X] : {new_x}, [time] : {end - start:.5f} sec\n")
  return new_x


def secant(f, x0, x1, tol=1e-6):

  global ITER
  global MAXITER

  if ITER is not 0 :
    ITER = 0

  print("-------------------------Secant Method-------------------------")
  print(f"[Starting Point] - [X0] : {x0}, [X1] : {x1} \n")

  start = time.time()

  while ITER < MAXITER:
      
      fx0 = f(x0)
      fx1 = f(x1)
  
      new_x = x1 - fx1 * ((x1 - x0) / (fx1 - fx0))
  
      ITER += 1
      print(f"[Root Finding...] - [iter] : {ITER}, [X] : {new_x} ")
  
      if (abs(new_x - x1)) < tol:
        break
  
      x0 = x1
      x1 = new_x

  end = time.time()
  print("\n=========================Root Found!=========================")
  print(f"[Terminal Result] - [iter] : {ITER}, [X] : {new_x}, [time] : {end - start:.5f} sec\n")
  return new_x

def regular_falsi(f, a, b, tol=1e-6):
  
  global ITER
  global MAXITER

  if ITER is not 0 :
    ITER = 0

  print("-------------------------Regular Falsi Method-------------------------")
  print(f"[Starting Point] - [a] : {a}, [b] : {b} \n")

  start = time.time()

  while ITER < MAXITER:

    fa = f(a)
    fb = f(b)

    new_x = b - fb * ((b - a) / (fb - fa))

    ITER += 1
    print(f"[Root Finding...] - [iter] : {ITER}, [X] : {new_x} ")

    if (abs(new_x - b)) < tol:
      break

    a = b
    b = new_x

  end = time.time()
  print("\n =========================Root Found!=========================")
  print(f"[Terminal Result] - [iter] : {ITER}, [X] : {new_x}, [time] : {end - start:.5f} sec\n")
  return new_x

