import numpy as np
import math

from algo import Nelder_Mead

f1 = lambda x,y : (x+3*y-5)**2 + (3*x + y - 7)**2
f2 = lambda x,y : 50*(x-y**2)**2 + (1-y)**2
f3 = lambda x,y : (1.5-x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def main():
  Nelder_Mead(f1)
  Nelder_Mead(f2)
  Nelder_Mead(f3)


if __name__ == '__main__':
  main()
  