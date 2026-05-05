import casadi as ca

class DeGrooteRigidTendonMuscle:

  def __init__(self, FM0, lM0, lTs, alpha0, vM_max, beta):
    self.FM0 = FM0
    self.lM0 = lM0
    self.lTs = lTs
    self.alpha0 = alpha0
    self.vM_max = vM_max
    self.beta = beta

    self.b11 = 0.8150671134243542
    self.b21 = 1.055033428970575
    self.b31 = 0.162384573599574
    self.b41 = 0.063303448465465
    self.b12 = 0.433004984392647
    self.b22 = 0.716775413397760
    self.b32 = -0.029947116970696
    self.b42 = 0.200356847296188
    self.b13 = 0.1
    self.b23 = 1.0
    self.b33 = 0.353553390593274
    self.b43 = 0.0

    self.k_pe = 4.0
    self.e0 = 0.6
    self.min_norm_fiber_length = 0.2

    self.d1 = -0.3211346127989808
    self.d2 = -8.149
    self.d3 = -0.374
    self.d4 = 0.8825327733249912

    self.lM = None
    self.lM_tilde = None
    self.cos_alpha = None
    self.vM = None
    self.vM_tilde = None
    self.f_act = None
    self.f_pas = None
    self.fv = None
    self.FM_iso = None
    self.F_CE = None
    self.FM = None
    self.FT = None

  def get_tendon_force(self, a, lMT, vMT):
    self.lM = ca.sqrt((lMT - self.lTs)**2 + (self.lM0 * ca.sin(self.alpha0))**2)
    self.lM_tilde = self.lM / self.lM0

    self.cos_alpha = (lMT - self.lTs) / self.lM

    self.vM = vMT * self.cos_alpha
    self.vM_tilde = self.vM / self.vM_max

    b1 = [self.b11, self.b12, self.b13]
    b2 = [self.b21, self.b22, self.b23]
    b3 = [self.b31, self.b32, self.b33]
    b4 = [self.b41, self.b42, self.b43]

    self.f_act = sum(b1[i] * ca.exp(-0.5 * (self.lM_tilde - b2[i])**2 / (b3[i] + b4[i] * self.lM_tilde)**2) for i in range(3))

    offset = ca.exp(self.k_pe * (self.min_norm_fiber_length - 1) / self.e0)
    self.f_pas = (ca.exp(self.k_pe * (self.lM_tilde - 1) / self.e0) - offset) / (ca.exp(self.k_pe) - offset)

    self.fv = self.d1 * ca.log(self.d2 * self.vM_tilde + self.d3 + ca.sqrt((self.d2 * self.vM_tilde + self.d3)**2 + 1)) + self.d4

    self.FM_iso = self.FM0 * a * self.f_act

    self.F_CE = self.FM0 * (a * self.f_act * self.fv + self.beta * self.vM_tilde)

    self.FM = self.FM0 * (a * self.f_act * self.fv + self.f_pas + self.beta * self.vM_tilde)

    self.FT = self.FM * self.cos_alpha

    return self.FT
