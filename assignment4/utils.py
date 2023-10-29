# import numpy as np

import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import jacobian

def gradient(f, x, y ) :
  h = 0.00001

  dx = (f([x + h, y]) - f([x - h, y])) / (2 * h)
  dy = (f([x, y + h]) - f([x, y - h])) / (2 * h)

  return np.array([dx, dy])


def bisection(f, a, b, tol=1e-6, MaxIter=10000):
  iter = 1

  if iter >= MaxIter:
      return (a+b)/2
  
  fx = f((a+b)/2)

  if (fx == 0.0) or ((b-a) < tol):
    iter += 1
    return (a+b)/2
  
  elif fx*f(a) < 0:
    b = (a+b)/2
    iter += 1
    return bisection(f, a, b, tol)
  
  else:
    a = (a+b)/2
    iter += 1
    return bisection(f, a, b, tol)
  
def inexact_line_search(f, x, y, dx, dy, c1=0.1, c2=0.5, beta=0.9):
  x_a = 1 # initial step size, 초기값에 관계 없이 특정값으로 수렴
  y_a = 1 # initial step size, 초기값에 관계 없이 특정값으로 수렴

  max_iter = 50
  iter = 1

  result = np.zeros(2)

  while iter <= max_iter:
        f_val = f(np.array([x + x_a * dx, y + y_a * dy]))
        original_f_val = f(np.array([x, y]))
        gradient_magnitude = gradient(f, x + x_a * dx, y + y_a * dy)
        original_gradient_magnitude = gradient(f, x, y)

        # Wolfe condition for x
        if f_val <= original_f_val + c1 * x_a * dx:
            if gradient_magnitude[0] * dx >= c2 * original_gradient_magnitude[0] * dx:
                x_a = x_a
        
        if f_val <= original_f_val + c1 * y_a * dx:
            if gradient_magnitude[0] * dx >= c2 * original_gradient_magnitude[0] * dx:
                y_a = y_a

        if result[0] == x_a and result[1] == y_a:
            return result
        else:
            result[0] = x_a
            result[1] = y_a

        x_a = beta * x_a
        y_a = beta * y_a

        iter += 1

  return np.array([x_a, y_a])
  
def hessian_matrix(f, x, y):
  
  hessian_f = jacobian(egrad(f))
  hessian = hessian_f(np.array([x,y]))
  return hessian

