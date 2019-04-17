from scipy.optimize import check_grad

def func(x):
  return x[0]**2 - 0.5 * x[1]**3
def grad(x):
  return [2 * x[0], -1.5 * x[1]**2]


print(check_grad(func, grad, [1.5, -1.5]))