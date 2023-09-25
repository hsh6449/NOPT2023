import numpy as np
import math

from search import *

f1 = lambda x: x ** 2 + 54 / x
f2 = lambda x: abs(x-0.65)

a, b = fibonacci(f2, 0, 1, 8)

print("a = ", a)
print("b = ", b)