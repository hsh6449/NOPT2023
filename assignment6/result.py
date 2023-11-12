from method import *
from utils import *

import numpy as np
import pandas as pd
import pdb

data_1 = pd.read_table("Data_for_Homework6/Model1_Data.txt", engine='python') # dataframe
data_2 = pd.read_table("Data_for_Homework6/Model2_Data.txt", engine='python')

data_1 = delete_nan(data_1)
data_2 = delete_nan(data_2)

print(data_1)
print(data_2)

def f1(x,y,z):

  # initial coefficient setting
  a = 1 
  b = 1
  c = 1
  d = 0

  return a*x + b*y + c*z + d

def f2(x,y,z, P):

  # initial coefficient setting
  a = P[0]
  b = P[1]
  c = P[2]
  d = p[4]

  return np.exp(((x-a)**2 + (y-b)**2 + (z-c)**2)/d**2)


# pdb.set_trace()



def main():
  pass

# if __name__ == "__main__":
#   main()

