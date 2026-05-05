import casadi as ca

class AndersonPandyLigament:

  def __init__(self, k0, k1, k2, k3, k4, theta, phi):
    self.k0 = k0
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3
    self.k4 = k4
    self.theta = theta
    self.phi = phi

    self.eta = 0.001

  def get_torque(self, q, v):
    T_lig = self.k0 + self.k1 * ca.exp(self.k2 * (q - self.theta)) + self.k3 * ca.exp(self.k4 * (q - self.phi))
    T_damp = -self.eta * v
    return T_lig + T_damp
