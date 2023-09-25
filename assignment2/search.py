import math
import numpy as np
import time

def fibonacci(f, a, b, n):
  fibonacci_numbers = [1, 1]
  
  for i in range(2, n + 1):
    fibonacci_numbers.append(fibonacci_numbers[i - 1] + fibonacci_numbers[i - 2]) # generate fibonacci numbers

  x1 = a + (b - a) * fibonacci_numbers[n - 2] / fibonacci_numbers[n] # choose x1 and x2
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
    else:
      b = x2
      x2 = x1
      f2 = f1
      x1 = a + (b - a) * fibonacci_numbers[n - i - 2] / fibonacci_numbers[n - i]
      f1 = f(x1)

  return (a,b)


def golden_section(f, a, b, TOL=1e-5):
  f1 = f(a)
  f2 = f(b)

  x1 = a + 0.618 * (b - a)
  x2 = b - 0.618 * (b - a)

  f1 = f(x1)
  f2 = f(x2)

  while abs(b - a) > TOL:
    if f1 > f2:
      a = x1
      x1 = x2
      f1 = f2
      x2 = b - 0.618 * (b - a)
      f2 = f(x2)
    else:
      b = x2
      x2 = x1
      f2 = f1
      x1 = a + 0.618 * (b - a)
      f1 = f(x1)

  return (a,b)
