import matplotlib.pyplot as plt
import numpy as np

import rootfinding as rf
from visuallize import plot

f1 = lambda x: x**3 - 3*x**2 + 3*x - 1 # x = 1
f2 = lambda x: x**5 - 4*x**4 + 3*x**3 + 9*x - 5 # x = 0.536
f3 = lambda x: np.exp(x) - x**np.exp(1) + 3*x**3 -2*x -5  # x = 1.249 

f_com = lambda x: x**3 + np.cos(x) # a = -1, b = 0

b_root, b_a, b_b = rf.bisection(f1, -100, 100)
n_root = rf.newton(f1, -300)
s_root = rf.secant(f_com, -1, 0)
r_root = rf.regular_falsi(f_com, -1, 0)

print(f"Bisection : root - {b_root:.3f}, [{b_a}, {b_b}]")
print(f"Newton : root - {n_root:.3f}")
print(f"Secant : root - {s_root:.3f}")
print(f"Regular Falsi : root - {r_root:.3f}")