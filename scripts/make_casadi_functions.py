import casadi as ca

def _define_variables(num_muscles, num_coordinates, x_list=None, a=None, q=None, v=None, u=None, d=None):
  if x_list is None:
    if a is None:
      a = ca.SX.sym('a', num_muscles)

    if q is None:
      q = ca.SX.sym('q', num_coordinates)

    if v is None:
      v = ca.SX.sym('v', num_coordinates)

    x_list = [a, q, v]

  if u is None:
    u = ca.SX.sym('u', num_muscles)

  if d is None:
    d = ca.SX.sym('d', num_coordinates)

  return x_list, u, d

def make_f(musculoskeletal_system, num_muscles, num_coordinates, x_list=None, a=None, q=None, v=None, u=None, d=None):
  x_list, u, _ = _define_variables(num_muscles, num_coordinates, x_list, a, q, v, u, d)

  x_dot_list = musculoskeletal_system.get_state_derivatives(x_list, u)

  x = ca.vertcat(*x_list)
  x_dot = ca.vertcat(*x_dot_list)
  L = ca.sum(musculoskeletal_system.E_dot)

  return ca.Function('f', [x, u], [x_dot, L], ['x', 'u'], ['x_dot', 'L'])

def make_f_grf_z(musculoskeletal_system, num_muscles, num_coordinates, contact_names, x_list=None, a=None, q=None, v=None, u=None, d=None):
  x_list, u, _ = _define_variables(num_muscles, num_coordinates, x_list, a, q, v, u, d)

  musculoskeletal_system.get_state_derivatives(x_list, u)

  x = ca.vertcat(*x_list)
  grf_z = sum(
    W_mobilized_body[2]
    for contact_name, W_mobilized_body in zip(contact_names, musculoskeletal_system.W_mobilized_body_list)
    if contact_name.endswith('_r')
  )

  return ca.Function('f_grf_z', [x], [grf_z], ['x'], ['grf_z'])

def make_A(musculoskeletal_system, num_muscles, num_coordinates, x_list=None, a=None, q=None, v=None, u=None, d=None):
  x_list, u, _ = _define_variables(num_muscles, num_coordinates, x_list, a, q, v, u, d)

  x_dot_list = musculoskeletal_system.get_state_derivatives(x_list, u)

  x = ca.vertcat(*x_list)
  x_dot = ca.vertcat(*x_dot_list)
  A = ca.jacobian(x_dot, x)

  return ca.Function('A', [x, u], [A], ['x', 'u'], ['A'])

def make_f_gait_data(musculoskeletal_system, num_muscles, num_coordinates, contact_names, x_list=None, a=None, q=None, v=None, u=None, d=None):
  x_list, u, _ = _define_variables(num_muscles, num_coordinates, x_list, a, q, v, u, d)

  musculoskeletal_system.get_state_derivatives(x_list, u)

  x = ca.vertcat(*x_list)
  tau = musculoskeletal_system.tau
  grf = sum(
    W_mobilized_body[:3]
    for contact_name, W_mobilized_body in zip(contact_names, musculoskeletal_system.W_mobilized_body_list)
    if contact_name.endswith('_r')
  )

  return ca.Function('f_gait_data', [x], [tau, grf], ['x'], ['tau', 'grf'])

def make_f_px_mobilized_body(musculoskeletal_system, num_muscles, num_coordinates, x_list=None, a=None, q=None, v=None, u=None, d=None):
  x_list, u, _ = _define_variables(num_muscles, num_coordinates, x_list, a, q, v, u, d)

  musculoskeletal_system.get_state_derivatives(x_list, u)

  x = ca.vertcat(*x_list)
  px_mobilized_body = ca.vertcat(*[musculoskeletal_system.T_mobilized_body_list[i][0, 3] for i in range(len(musculoskeletal_system.T_mobilized_body_list))])

  return ca.Function('f_px_mobilized_body', [x], [px_mobilized_body], ['x'], ['px_mobilized_body'])

def make_f_sim(musculoskeletal_system, num_muscles, num_coordinates, x_list=None, a=None, q=None, v=None, u=None, d=None):
  x_list, u, d = _define_variables(num_muscles, num_coordinates, x_list, a, q, v, u, d)

  x_dot_list = musculoskeletal_system.get_state_derivatives(x_list, u, d)

  x = ca.vertcat(*x_list)
  x_dot = ca.vertcat(*x_dot_list)

  return ca.Function('f_sim', [x, u, d], [x_dot], ['x', 'u', 'd'], ['x_dot'])
