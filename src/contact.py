import casadi as ca

class SmoothHuntCrossleyContact:

  def __init__(self, p_sphere_in_mobilized_body, R, T_half_space, E=1.0, c=0.0, us=0.0, ud=0.0, uv=0.0, vt=0.01, cf=1e-5, bd=300.0, bv=50.0):
    self.p_sphere_in_mobilized_body = p_sphere_in_mobilized_body
    self.R = R
    self.T_half_space = T_half_space
    self.E = E
    self.c = c
    self.us = us
    self.ud = ud
    self.uv = uv
    self.vt = vt
    self.cf = cf
    self.bd = bd
    self.bv = bv

  def get_force(self, T_mobilized_body, nu_mobilized_body_in_local):
    R_mobilized_body = T_mobilized_body[:3, :3]
    p_mobilized_body = T_mobilized_body[:3, 3]

    v_mobilized_body_in_local = nu_mobilized_body_in_local[:3]
    omega_mobilized_body_in_local = nu_mobilized_body_in_local[3:]

    p_sphere = p_mobilized_body + R_mobilized_body @ self.p_sphere_in_mobilized_body
    v_sphere = R_mobilized_body @ (v_mobilized_body_in_local + ca.cross(omega_mobilized_body_in_local, self.p_sphere_in_mobilized_body))
    omega_sphere = R_mobilized_body @ omega_mobilized_body_in_local

    R_half_space = self.T_half_space[:3, :3]
    p_half_space = self.T_half_space[:3, 3]

    normal = R_half_space @ ca.DM([1, 0, 0])

    p_sphere_wrt_half_space = p_sphere - p_half_space
    x = ca.dot(p_sphere_wrt_half_space, normal) + self.R

    p_contact = p_sphere + (self.R - 0.5 * x) * normal

    p_contact_wrt_sphere = p_contact - p_sphere
    v_contact = v_sphere + ca.cross(omega_sphere, p_contact_wrt_sphere)

    v = ca.dot(v_contact, normal)
    v_contact_normal = v * normal

    vtangent = v_contact - v_contact_normal

    k = 0.5 * self.E**(2 / 3)

    fh_pos = 4 / 3 * k * ca.sqrt(self.R * k) * ca.sqrt(x**2 + self.cf)**1.5

    fh_smooth = fh_pos * (0.5 + 0.5 * ca.tanh(self.bd * x))

    fhc_pos = fh_smooth * (1 + 1.5 * self.c * v)

    fhc_smooth = fhc_pos * (0.5 + 0.5 * ca.tanh(self.bv * (v + 2 / (3 * self.c))))

    vs = ca.sqrt(ca.sumsqr(vtangent) + self.cf)

    ff = fhc_smooth * ca.fmin(vs / self.vt, 1) * (self.ud + 2 * (self.us - self.ud) / (1 + (vs / self.vt)**2) + self.uv * vs)

    force = fhc_smooth * normal + ff * vtangent / vs

    reaction_force = -force

    p_contact_wrt_mobilized_body = p_contact - p_mobilized_body
    m_mobilized_body = ca.cross(p_contact_wrt_mobilized_body, reaction_force)

    W_mobilized_body = ca.vertcat(reaction_force, m_mobilized_body)

    return W_mobilized_body
