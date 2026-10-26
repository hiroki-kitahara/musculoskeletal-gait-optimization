import casadi as ca

from msk.utils import smooth_step_function

class BhargavaMetabolicEnergy:

  def __init__(self, m, f_slow):
    self.m = m
    self.f_slow = f_slow

    self.A_fast_dot = 133.0
    self.A_slow_dot = 40.0
    self.M_fast_dot = 111.0
    self.M_slow_dot = 74.0

  def get_energy_consuption_rate(self, u, lM_tilde, v_CE, FM_iso, F_CE, FM):
    u_fast = 1 - ca.cos(ca.pi / 2 * u)
    u_slow = ca.sin(ca.pi / 2 * u)

    A_dot = self.m * (1 - self.f_slow) * self.A_fast_dot * u_fast + self.m * self.f_slow * self.A_slow_dot * u_slow

    M_dot = lM_tilde * self.m * (1 - self.f_slow) * self.M_fast_dot * u_fast + lM_tilde * self.m * self.f_slow * self.M_slow_dot * u_slow

    alpha_con = 0.16 * FM_iso + 0.18 * FM
    alpha_ec = 0.157 * FM
    alpha = alpha_con * smooth_step_function(-v_CE) + alpha_ec * smooth_step_function(v_CE)
    S_dot = -alpha * v_CE

    W_dot = -F_CE * v_CE * smooth_step_function(-v_CE)

    H_dot = A_dot + M_dot + S_dot

    H_dot_per_mass = H_dot / self.m
    H_dot_per_mass_clamped = H_dot_per_mass * smooth_step_function(H_dot_per_mass - 1.0) + 1.0 * smooth_step_function(1.0 - H_dot_per_mass)
    H_dot_clamped = H_dot_per_mass_clamped * self.m

    return H_dot_clamped + W_dot
