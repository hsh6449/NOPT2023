import matplotlib.pyplot as plt
import rootfinding as rf

f1 = lambda x: x**3 - 3*x**2 + 3*x - 1
f2 = lambda x: x**5 - 4*x**4 + 3*x**3 + 9*x - 5

b_root, b_a, b_b = rf.bisection(f2, -15, 17)

print(f"Bisection : root - {b_root:.3f}, [{b_a}, {b_b}]")