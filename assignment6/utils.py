import numpy as np
import autograd.numpy as np

import pdb

def jaccobian(f, x, y, z, P ) :
  h = 0.00001
  
  jaco = []
  temp_p = np.copy(P)

  for i in range(len(P)):
    temp_p[i] = temp_p[i] + h

    fval = f(x,y,z, temp_p) 
    dP = (fval- f(x,y,z, P)) / h

    jaco.append(dP)
    temp_p = np.copy(P)

  jaco = np.stack(np.array(jaco), axis=0)

  return jaco
  
def gradient(f, x, y ) :
  h = 0.00001

  dx = (f([x + h, y]) - f([x - h, y])) / (2 * h)
  dy = (f([x, y + h]) - f([x, y - h])) / (2 * h)

  return np.array([dx, dy])
  

def delete_nan(data):
  if (data.iloc[0].isnull().values.any() == True):
    data = data.dropna(axis=0)
    data = data.reset_index(drop=True)
  return data