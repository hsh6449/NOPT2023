import matplotlib.pyplot as plt
import rootfinding as rf

import numpy as np

f1 = lambda x: x**3 - 3*x**2 + 3*x - 1
f2 = lambda x: x**5 - 4*x**4 + 3*x**3 + 9*x - 5
f3 = lambda x: x**3 + np.cos(x) # a = -1, b = 0

b_root, b_a, b_b = rf.bisection(f2, -15, 17)
n_root = rf.newton(f2, 3)
s_root = rf.secant(f2, -10, 15)
r_root = rf.regular_falsi(f2, -10, 15)

print(f"Bisection : root - {b_root:.3f}, [{b_a}, {b_b}]")
print(f"Newton : root - {n_root:.3f}")
print(f"Secant : root - {s_root:.3f}")
print(f"Regular Falsi : root - {r_root:.3f}")
