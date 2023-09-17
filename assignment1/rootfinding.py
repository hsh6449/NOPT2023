import numpy as np
import math
import time

MAXITER = 10000
ITER = 0

def bisection(f, a, b, tol=1e-6):

  global ITER
  global MAXITER

  result = []

  if ITER == 0 :
    print("-------------------------Bisection Method-------------------------")
    print(f"[Starting Point] - [a] : {a}, [b] : {b} \n")
  
  start = time.time()

  if ITER == MAXITER:
      return (a+b)/2, a, b
  
  fx = f((a+b)/2)

  if (fx == 0.0) or ((b-a) < tol):
    ITER += 1
    end = time.time()


    # if fx is not 0.0:
    #   print("Root in not here... Try Agian with wider range!")
    #   return (a+b)/2, a, b

    # else :
    print("=========================Root Found!=========================")
    print(f"[Terminal Result] - [iter] : {ITER}, [a] : {a}, [b] : {b}, [time] : {end-start:.6f} sec\n")

    result.append((a+b)/2)
    
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


def newton(f, x0, tol=1e-6):

  global ITER
  global MAXITER

  if ITER is not 0 :
    ITER = 0

  h = 1e-6
  old_x = x0

  print("-------------------------Newton Method-------------------------")
  print(f"[Starting Point] - [X] : {old_x} \n")
  start = time.time()

  while ITER < MAXITER:

    fx = f(old_x)
    dfx = (f(old_x + h) - f(old_x)) / h

    new_x = old_x - fx/(dfx + h) # h is added to prevent zero division

    ITER += 1
    print(f"[Root Finding...] - [iter] : {ITER}, [X] : {new_x} ")

    if (f(new_x) == 0 ) or ((abs(new_x - old_x)) < tol):
      print("break!")
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
        print("break!")
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

  prev_x = 0

  print("-------------------------Regular Falsi Method-------------------------")
  print(f"[Starting Point] - [a] : {a}, [b] : {b} \n")

  start = time.time()

  while ITER < MAXITER:

    fa = f(a)
    fb = f(b)
    
    new_x = b - fb * ((b - a) / (fb - fa))

    if fa * f(new_x) < 0:
      b = new_x
    else:
      a = new_x

    ITER += 1
    print(f"[Root Finding...] - [iter] : {ITER}, [a] : {a}, [b] : {b}")

    if abs(new_x - prev_x) < tol:
      break

    prev_x = new_x


  end = time.time()
  print("\n =========================Root Found!=========================")
  print(f"[Terminal Result] - [iter] : {ITER}, [a] : {a}, [b] : {b}, [time] : {end - start:.5f} sec\n")
  return new_x

