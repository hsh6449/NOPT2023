import numpy as np
import math

from search import *

f1 = lambda x: x**3 - 3*x**2 + 3*x - 1 # x = 1
f2 = lambda x: x**5 - 4*x**4 + 3*x**3 + 9*x - 5 # x = 0.536
f3 = lambda x: np.exp(x) - x**np.exp(1) + 3*x**3 -2*x -5  # x = 1.249 
f4 = lambda x: -20*x*np.exp(-0.2*np.square(x**2)) -np.exp(0.5*(np.cos(2*np.pi*x) + 1)) + 20 + np.exp(1) # x = 0

# a, b = fibonacci(f4, -5, 5, 15)
function_list = [f1, f2, f3, f4]
for f in function_list:
  print(f"Function : f_{function_list.index(f) + 1}")
  print("-----------------------------------\n")
  a, b = fibonacci(f, -5, 5, 15)

for f in function_list:
  print(f"Function : f_{function_list.index(f) + 1}")
  print("-----------------------------------\n")
  a, b = golden_section(f4, -5, 5, 1e-5)