import math
import numpy as np
import time

def fibonacci(f, a, b, n):
  fibonacci_numbers = [1, 1]
  
  for i in range(2, n + 1):
    fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2]) # generate fibonacci numbers

  print("Fibonacci number list : ",fibonacci_numbers)
  print("-----------------------------------\n")

  print("Start point : [{}, {}]\n".format(a, b))

  start = time.time()

  x1 = a + (b - a) * fibonacci_numbers[n - 2] / fibonacci_numbers[n] # calculate x1 and x2
  x2 = b - (b - a) * fibonacci_numbers[n - 2] / fibonacci_numbers[n]

  f1 = f(x1) # calculate f(x1) and f(x2)
  f2 = f(x2)

  for i in range(1, n - 1): # eliminate the interval step
    if f1 > f2:
      a = x1
      x1 = x2
      f1 = f2
      x2 = b - (b - a) * fibonacci_numbers[n - i - 1] / fibonacci_numbers[n - i]
      f2 = f(x2)
      
      print(f"[Root finding...] - iter : {i} - interval : [{a}, {b}]" )
    else:
      b = x2
      x2 = x1
      f2 = f1
      x1 = a + (b - a) * fibonacci_numbers[n - i - 2] / fibonacci_numbers[n - i]
      f1 = f(x1)
      print(f"[Root finding...] - iter : {i} - interval : [{a}, {b}]" )
  end = time.time()

  print("-----------------------------------\n")
  print("End point : [{}, {}]\n".format(a, b))
  print("Time : ", end - start)

  return (a,b)


def golden_section(f, a, b, TOL=1e-5):

  print("Golden section method")
  print("-----------------------------------\n")
  print("Start point : [{}, {}]\n".format(a, b))

  start = time.time()
  x1 = a + 0.618 * (b - a) # 0.618 is the golden ratio
  x2 = b - 0.618 * (b - a)

  f1 = f(x1)
  f2 = f(x2)

  iter = 0
  while abs(b - a) > TOL:
    iter += 1
    if f1 > f2:
      a = x1
      x1 = x2
      f1 = f2
      x2 = b - 0.618 * (b - a)
      f2 = f(x2)
      print(f"[Root finding...] - iter : {iter} - interval : [{a}, {b}]" )
    else:
      b = x2
      x2 = x1
      f2 = f1
      x1 = a + 0.618 * (b - a)
      f1 = f(x1)
      print(f"[Root finding...] - iter : {iter} - interval : [{a}, {b}]" )
  
  end = time.time()
  print("-----------------------------------\n")
  print("End point : [{}, {}]\n".format(a, b))
  print("Time : ", end - start)

  return (a,b)
