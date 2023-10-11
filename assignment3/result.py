import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from algo import Nelder_Mead, powell

f1 = lambda x,y : (x+3*y-5)**2 + (3*x + y - 7)**2
f2 = lambda x,y : 50*(x-y**2)**2 + (1-y)**2
f3 = lambda x,y : (1.5-x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def contour_plot(result: list, values: list, jitter_amount=1e-10):
    plt.style.use('default')

    x = [i[0] + np.random.uniform(-jitter_amount, jitter_amount) for i in result]
    y = [i[1] + np.random.uniform(-jitter_amount, jitter_amount) for i in result]
    z = values

    if len(set(x)) == 1:
      print("All x values are the same. Cannot plot a contour.")
      return

    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    xi = np.linspace(min(x) - 0.1 * x_range, max(x) + 0.1 * x_range, 100)
    yi = np.linspace(min(y) - 0.1 * y_range, max(y) + 0.1 * y_range, 100)

    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    fig, ax = plt.subplots()
    cp = ax.contour(xi, yi, zi, colors='black')
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(x, y, color='red', marker='o')  # Mark the original points

    for i in range(1, len(x)):
      ax.arrow(x[i-1], y[i-1], x[i]-x[i-1], y[i]-y[i-1], head_width=0.5, head_length=1, fc='blue', ec='blue')

    plt.show()

def main(method = "Nelder_Mead", f = f2, random_seed=777):
  if method == "Nelder_Mead":
    result, values = Nelder_Mead(f, random_seed= random_seed)
    contour_plot(result, values)
  elif method == "powell":
    result, values = powell(f, random_seed= random_seed)
    contour_plot(result, values)
  else:
    print("No such method")
    pass
  


if __name__ == '__main__':
  main(method = "Nelder_Mead", f = f2, random_seed=777)
