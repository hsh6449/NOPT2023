import numpy as np
import random
import time

# function list to dict to list With sorting
def l2d2l(keys,values):
  """temp_dict = {keys[i] : values[i] for i in range(N+1)}

  sorted_fx_= dict(sorted(temp_dict.items(), key=lambda item: item[1])) # sort

  key = list(sorted_fx_.keys())
  value = list(sorted_fx_.values())"""

  combined = list(zip(keys, values))
  combined.sort(key=lambda kv: kv[1])
  key, value = zip(*combined)

  return list(key), list(value)


def Nelder_Mead(f, N=3, alpha=0.01, beta=1.5, gamma=0.05, epsilon=0.01, max_iter=10000000, random_seed=0):
  # step 0 : initialization
  random.seed(random_seed)
  start = time.time()

  x = []
  y = []
  coordinates = []
  results = []

  iter = 0

  for i in range(N+1):
    x.append(random.randint(-1000, 1000)) # 랜덤선택 전략 고민, 일단 N+1개 만들어줌 즉 indext는 0~N
    y.append(random.randint(-1000, 1000)) # 랜던선택 전략 고민

  fx = {(x[i],y[i]) : f(x[i], y[i]) for i in range(N+1)}
  sorted_fx_= dict(sorted(fx.items(), key=lambda item: item[1])) # sort

  keys_ = list(sorted_fx_.keys())
  values_ = list(sorted_fx_.values())

  prev_fx = values_[N]
  
  while iter < max_iter:
    if iter >= 1 :
      print(f"iter  - {iter}, (x, y) : {keys_[N]}, fxy : {values_[N]}\n")    

    iter += 1

    c_x, c_y = sum([k[0] for k in keys_[:-1]])/N, sum([k[1] for k in keys_[:-1]])/N

    # step 1 : reflection
    xr = c_x + alpha*(c_x - keys_[N][0]) # x 값 N+1번째 값이 index N
    yr = c_y + alpha*(c_y - keys_[N][1]) # y 값

    fr = f(xr, yr) # evalueate fr at xr, yr
  
    if  values_[0] <= fr <= values_[N-1] : 

      keys_[N] = (xr, yr)
      values_[N] = fr

    elif fr > values_[N-1]: # step 3 : contraction
      if fr < values_[N]:
        xc = c_x + gamma*(xr - c_x)
        yc = c_y + gamma*(yr - c_y)

      elif fr >= values_[N]:
        xc = c_x + gamma*(keys_[N][0] - c_x)
        yc = c_y + gamma*(keys_[N][1] - c_y)


      fc = f(xc, yc)
      min_value = np.min([fr, values_[N]])

      if fc < min_value:

        keys_[N] = (xc, yc)
        values_[N] = fc

      else: # fc >= min_value, step 4 : shrink
        for i in range(N+1):
          keys_[i] = (keys_[0][0] + 0.5*(keys_[i][0] - keys_[0][0]), keys_[0][1] + 0.5*(keys_[i][1] - keys_[0][1]))
          values_[i] = f(keys_[i][0], keys_[i][1])
          # keys_[i] = ( (keys_[i][0] + keys_[i][0]) / 2, (keys_[i][1] + keys_[i][1]) / 2 )
          # values_[i] = f(keys_[i][0], keys_[i][1])

    else : # fr <= values_[0]: # step 2 : expansion
      xe = c_x + beta*(xr - c_x)
      ye = c_y + beta*(yr - c_y)
      fe = f(xe, ye)

      if fe <= fr:

        keys_[N] = (xe, ye)
        values_[N] = fe

      else:
        
        keys_[N] = (xr, yr)
        values_[N] = fr
        
    keys_, values_ = l2d2l(keys_, values_)

    coordinates.append(keys_[N])
    results.append(values_[N])

    if abs(prev_fx - values_[N]) < epsilon:
      print(f"Converged after {iter} iterations.")
      break
      
        
    prev_fx = values_[N]
  
  if iter == max_iter:
    print(f"Reached max iterations without convergence.\n")

  end = time.time()
  print(f"Final result : (x, y) : {keys_[N]}, fxy : {values_[N]}\n, time : {end-start}")

  return coordinates, results


def powell(f, gamma=0.05, epsilon=0.01, max_iter=10000000, random_seed=0):

  random.seed(random_seed)

  x, y = random.randint(-1000, 1000), random.randint(-1000, 1000) # random initial point
  U = [(1,0), (0,1)] # initial direction is standard unit vector

  iter = 0
  start = time.time()

  coordinates = []
  results = []

  prev_fx = f(x, y)
  P = [(x, y)]

  while iter < max_iter:
    x_start , y_start = x, y

    coordinates.append((x, y))
    results.append(prev_fx)

    for u in U:
      # set the new x,y candidates
      x_new = x + gamma*u[0]
      y_new = y + gamma*u[1]

      # fx_new = f(x_new, y_new)
      
      gamma_f = lambda gamma : f(x_new + gamma*u[0], y_new + gamma*u[1])

      a, b = Univariate_search(gamma_f, initial_point=0, delta=0.01)
      optimal_gamma = golden_section(gamma_f, a, b)

      x, y = x_new + optimal_gamma*u[0], y_new + optimal_gamma*u[1]

      P.append((x,y))

    U.pop(0)
    U.append((x - x_start, y - y_start))


    iter += 1


    new_fx = f(x, y)
    print(f"iter  - {iter}, (x, y) : {(x, y)}, fxy : {new_fx}\n")

    # 종료조건
    if abs(prev_fx - new_fx) < epsilon:
      print(f"Converged after {iter} iterations.")
      break

    prev_fx = new_fx

  if iter == max_iter:
    print(f"Reached max iterations without convergence.\n")
  
  end = time.time()
  print(f"Final result : (x, y) : {(x, y)}, fxy : {f(x, y)}\n, time : {end-start}")

  return coordinates, results


def golden_section(f, a, b, TOL=1e-5):

  # assignment 2 에서 사용한 golden section method를 사용하여 powell 방법의 step size를 구하려 함
  # 코드는 assignment 2에서 가져옴 

  x1 = a + 0.618 * (b - a) # 0.618 is the golden ratio
  x2 = b - 0.618 * (b - a)

  f1 = f(x1) #c
  f2 = f(x2) #d

  while abs(b - a) > TOL:
    if f1 > f2:
      a = x1
      x1 = x2
      f1 = f2
      x2 = b - 0.618 * (b - a)
      f2 = f(x2)

    else:
      b = x2
      x2 = x1
      f2 = f1
      x1 = a + 0.618 * (b - a)
      f1 = f(x1)
  

  return (a + b)/2 

def Univariate_search(f, initial_point=0.0, delta=0.01):
    
    a, fa = initial_point, f(initial_point)
    b, fb = a + delta, f(a + delta)
    
    if fb > fa:
        a, b = b, a
        fa, fb = fb, fa
        delta = -delta

    while True:
        c, fc = b + delta, f(b + delta)
        if fc > fb:
            if a < c:
                return (a, c)
            else:
                return (c, a)
        a, fa, b, fb = b, fb, c, fc
        delta *= 2.0
