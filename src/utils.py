import casadi as ca

def smooth_step_function(x, b=10.0):
  return 0.5 + 0.5 * ca.tanh(b * x)
