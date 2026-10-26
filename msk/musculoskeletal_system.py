import casadi as ca

class MusculoskeletalSystem:

  def __init__(self, actuated_coordinate_indices, muscle_activation_dynamics, muscle_kinematics, muscle, ligament, contact_kinematics_functions, contacts, skeletal_dynamics_function, metabolic_energy, damping):
    self.actuated_coordinate_indices = actuated_coordinate_indices
    self.muscle_activation_dynamics = muscle_activation_dynamics
    self.muscle_kinematics = muscle_kinematics
    self.muscle = muscle
    self.ligament = ligament
    self.contact_kinematics_functions = contact_kinematics_functions
    self.contacts = contacts
    self.skeletal_dynamics_function = skeletal_dynamics_function
    self.metabolic_energy = metabolic_energy
    self.damping = damping

    self.lMT = None
    self.vMT = None
    self.MA = None
    self.FT = None
    self.T_lig = None
    self.tau = None
    self.T_mobilized_body_list = None
    self.W_mobilized_body_list = None
    self.E_dot = None
    self.x_dot = None

  def get_state_derivatives(self, x, u, d=0.0):
    a = x[0]
    q = x[1]
    v = x[2]

    q_actuated = q[self.actuated_coordinate_indices]
    v_actuated = v[self.actuated_coordinate_indices]

    a_dot = self.muscle_activation_dynamics.get_activation_derivative(a, u)

    self.lMT, self.vMT, self.MA = self.muscle_kinematics.get_kinematics(q_actuated, v_actuated)

    self.FT = self.muscle.get_tendon_force(a, self.lMT, self.vMT)

    self.T_lig = self.ligament.get_torque(q_actuated, v_actuated)

    self.tau = self.MA @ self.FT + self.T_lig - self.damping * v_actuated

    tau_total = ca.SX.zeros(v.shape[0])

    tau_total[self.actuated_coordinate_indices] = self.tau

    self.T_mobilized_body_list = []
    self.W_mobilized_body_list = []

    for contact_kinematics_function, contact in zip(self.contact_kinematics_functions, self.contacts):
      T_mobilized_body, nu_mobilized_body_in_local, J_mobilized_body = contact_kinematics_function(q, v)

      W_mobilized_body = contact.get_force(T_mobilized_body, nu_mobilized_body_in_local)

      tau_total += J_mobilized_body.T @ W_mobilized_body

      self.T_mobilized_body_list.append(T_mobilized_body)
      self.W_mobilized_body_list.append(W_mobilized_body)

    tau_total += d

    v_dot = self.skeletal_dynamics_function(q, v, tau_total)

    self.E_dot = self.metabolic_energy.get_energy_consuption_rate(u, self.muscle.lM_tilde, self.muscle.vM, self.muscle.FM_iso, self.muscle.F_CE, self.muscle.FM)

    self.x_dot = [a_dot, v, v_dot]

    return self.x_dot
