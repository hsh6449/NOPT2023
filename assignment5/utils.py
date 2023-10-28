# import numpy as np

# import autograd.numpy as np
# from autograd import elementwise_grad as egrad
# from autograd import jacobian

def gradient(f, x, y ) :
  h = 0.00001

  dx = (f([x + h, y]) - f([x - h, y])) / (2 * h)
  dy = (f([x, y + h]) - f([x, y - h])) / (2 * h)

  return dx, dy


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
  
def inexact_line_search(f, x, y, dx, dy, c=0.3, beta=0.2):
  a = 1 # initial step size
  iter = 1
  while f([x + a * dx, y + a * dy]) > f([x, y]) + c * a * (dx**2 + dy**2):
    a = beta * a
    iter += 1

  print(iter)
  
  return a
  
# def hessian_matrix(f, x, y):
  
#   hessian_f = jacobian(egrad(f))
#   hessian = hessian_f(np.array([x,y]))
#   return hessian

