import matplotlib.pyplot as plt
import numpy as np

from method import *
from scipy.interpolate import griddata

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

def contour_plot(result: list, values: list, jitter_amount=1e-8):
    plt.style.use('default')

    x = [i[0] + np.random.uniform(-jitter_amount, jitter_amount) for i in result]
    y = [i[1] + np.random.uniform(-jitter_amount, jitter_amount) for i in result]
    z = values

    if len(set(x)) == 1:
        print("All x values are the same. Cannot plot a contour.")
        return

    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    # xi = np.linspace(min(x), max(x), 100)
    # yi = np.linspace(min(y), max(y), 100)
    xi = np.linspace(min(x) - 0.05 * x_range, max(x) + 0.05 * x_range, 10)
    yi = np.linspace(min(y) - 0.05 * y_range, max(y) + 0.05 * y_range, 10)

    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    # If NaN values are present, use 'nearest' interpolation method
    if np.isnan(zi).any():
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='nearest')

    fig, ax = plt.subplots()
    cp = ax.contour(xi, yi, zi, colors='black')
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(x, y, color='red', marker='o')  # Mark the original points

    arrow_head_width = x_range * 0.005
    arrow_head_length = y_range * 0.0075

    if x_range < 1e-6:
       pass
    else:
      for i in range(1, len(x)):
          ax.arrow(x[i-1], y[i-1], x[i]-x[i-1], y[i]-y[i-1], head_width=arrow_head_width, head_length=arrow_head_length, fc='blue', ec='blue')

    plt.show()

def main(f, x=1.2, y=1.2, epsilon=0.0000001):
  # coordinates, results = steepest_descent(f, x, y,epsilon)
  # contour_plot(coordinates, results)
  # coordinates, results = newton(f, x, y,epsilon)
  # contour_plot(coordinates, results)
  coordinates, results = SR1(f, x, y,epsilon)
  contour_plot(coordinates, results)
  # coordinates, results = BFGS(f, x, y,epsilon)
  # contour_plot(coordinates, results)



if __name__ == "__main__":
  main(f3, x = 16.0, y = 100.0)
  # main(f2)
  # main(f3)