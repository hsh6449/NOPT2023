from method import *

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


def main(f, x0, max_iter=10000, eps=1e-5, verbose=False, mode = "cg_fr"):
  print(f"Start Finding {f.__name__} minimum")

  if mode == "linear_conjugate":
    x, y = linear_conjugate(f = f, x0=x0, verbose=True)
  if mode == "cg_fr":
    x, y = cg_fr(f = f, x0=x0, verbose=True)
  if mode == "cg_pr":
    x, y = cg_pr(f = f, x0=x0, verbose=True)
  if mode == "cg_hs":
    x, y = cg_hs(f = f, x0=x0, verbose=True) 

  print(x, y)

if __name__ == "__main__":
  main(f = f3, x0=[0.5,0.5], max_iter=10000, eps=1e-5, verbose=True, mode = "cg_fr") 