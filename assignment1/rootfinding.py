import numpy as np
import math
import time

MAXITER = 100
ITER = 0

def bisection(f, a, b, tol=1e-6):

  global ITER
  global MAXITER
  start = time.time()

  if ITER == MAXITER:
      return a, b
  
  fx = f((a+b)/2)

  if (fx == 0) or ((b-a) < tol):
    ITER += 1
    end = time.time()

    print("=========================Root Found!=========================")
    print(f"[terminal interval] - [iter] : {ITER}, [a] : {a}, [b] : {b}, time : {end-start:.9f} sec")

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


def newton(f, df, x0, tol=1e-6):

  global ITER
  global MAXITER
  pass

def secant(f, x0, x1, tol=1e-6, maxiter=100):
  pass

def regular_falsi(f, a, b, tol=1e-6, maxiter=100):
  pass