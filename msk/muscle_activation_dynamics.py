from msk.utils import smooth_step_function

class RaaschMuscleActivationDynamics:

  def __init__(self):
    self.tau_a = 0.011
    self.tau_d = 0.068

  def get_activation_derivative(self, a, u):
    a_dot_act = (u - a) * (u / self.tau_a + (1 - u) / self.tau_d)
    a_dot_deact = (u - a) / self.tau_d
    return a_dot_act * smooth_step_function(u - a) + a_dot_deact * smooth_step_function(a - u)
