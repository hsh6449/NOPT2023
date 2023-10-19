import matplotlib.pyplot as plt
import numpy as np

from method import *

# f1 = lambda x,y : (x+3*y-5)**2 + (3*x + y - 7)**2
# f2 = lambda x,y : 50*(x-y**2)**2 + (1-y)**2
# f3 = lambda x,y : (1.5-x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def f1(x):
  y = x[1]
  x = x[0]
  return (x+3*y-5)**2 + (3*x + y - 7)**2

def f2(x):
  y = x[1]
  x = x[0]
  return 50*(x-y**2)**2 + (1-y)**2

def f3(x):
  y = x[1]
  x = x[0]
  return (1.5-x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

x1, y1 = steepest_descent(f3, 1.2, 1.2)
x1, y1 = newton(f1, 1.2, 1.2, 1e-6, 10000)
x1, y1 = SR1(f1, 1.2, 1.2, 1e-6, 10000)