from method import *
from utils import *

import numpy as np
import pandas as pd
import pdb

model1 = pd.read_table("Data/Model1_Data.txt", engine='python') # dataframe
model2 = pd.read_table("Data/Model2_Data.txt", engine='python')

model1 = delete_nan(model1)
model2 = delete_nan(model2)


def f1(x,y,z, P):

  # initial coefficient setting
  a = P[0]
  b = P[1]
  c = P[2]
  d = P[3]

  return a*x + b*y + c*z + d

def f2(x,y,z, P):

  # initial coefficient setting
  a = P[0]
  b = P[1]
  c = P[2]
  d = P[3]

  return np.exp(((x-a)**2 + (y-b)**2 + (z-c)**2)/d**2)

def main():
  # Gauss_Newton(f1, model1, 1)
  # Gauss_Newton(f2, model1, 2)
  Levenberg_Marquardt(f2, model2, 1)
  Levenberg_Marquardt(f2, model2, 2)

if __name__ == "__main__":
  main()

