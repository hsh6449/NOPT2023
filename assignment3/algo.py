import numpy as np
import random
import math

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


def Nelder_Mead(f, N=3, alpha=0.01, beta=1.5, gamma=0.05, epsilon=0.001, max_iter=10000000):
  # step 0 : initialization
  x = []
  y = []

  iter = 0

  for i in range(N+1):
    x.append(random.randint(-1000, 1000)) # 랜덤선택 전략 고민, 일단 N+1개 만들어줌 즉 indext는 0~N
    y.append(random.randint(-1000, 1000)) # 랜던선택 전략 고민

  fx = {(x[i],y[i]) : f(x[i], y[i]) for i in range(N+1)}
  sorted_fx_= dict(sorted(fx.items(), key=lambda item: item[1])) # sort

  keys_ = list(sorted_fx_.keys())
  values_ = list(sorted_fx_.values())

  prev_fx = values_[N]
  fr = 0
  
  while iter < max_iter:
    if iter >= 1 :
      print(f"iter  - {iter}, (x, y) : {keys_[N]}, fxy : {values_[N]}\n")    

    iter += 1

    c_x, c_y = sum([k[0] for k in keys_[:-1]])/N, sum([k[1] for k in keys_[:-1]])/N

    # step 1 : reflection
    xr = c_x + alpha*(c_x - keys_[N][0]) # x 값 N+1번째 값이 index N
    yr = c_y + alpha*(c_y - keys_[N][1]) # y 값

    fr = f(xr, yr) # evalueate fr at xr, yr
  
    if  values_[0] <= fr <= values_[N-1] : # f(xr, yr) >= f(x1, y1) and f(xr, yr) <= f(xn, yn)
      # prev_fx = values_[N]

      keys_[N] = (xr, yr)
      values_[N] = fr

      # keys_, values_ = l2d2l(keys_, values_, N)

      # prev_fx = new_fx
      # new_fx = fr
      # print(f"iter  - {iter}, xr, yr : {xr}, {yr}")

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
        # prev_fx = values_[N]

        keys_[N] = (xc, yc)
        values_[N] = fc

        # keys_, values_ = l2d2l(keys_, values_, N)

        # prev_fx = new_fx
        # new_fx = fc
        # print(f"iter  - {iter}, xc, yc : {xc}, {yc}")

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
        # prev_fx = values_[N]

        keys_[N] = (xe, ye)
        values_[N] = fe
        
        # prev_fx = new_fx
        # new_fx = fe
        # print(f"iter  - {iter}, xe, ye : {xe}, {ye}")
      else:
        # prev_fx = values_[N]

        keys_[N] = (xr, yr)
        values_[N] = fr

        # keys_, values_ = l2d2l(keys_, values_, N)

        # prev_fx = new_fx
        # new_fx = fr
        # print(f"iter  - {iter}, xr, yr : {xr}, {yr}")

    keys_, values_ = l2d2l(keys_, values_)

    if abs(prev_fx - values_[N]) < epsilon:
      print(f"Converged after {iter} iterations.")
      break
      # return keys_[N], values_[N]
        
    prev_fx = values_[N]

  print(f"Reached max iterations without convergence.")

  # return keys_[N], values_[N]

def powell(f, N=10, gamma=0.05, epsilon=0.001, max_iter=10000000):
  
  p = [(1,0), (0,1)] # initial direction is standard unit vector

  for k in range(N):
    pass

  pass
