import pickle

import casadi as ca
import numpy as np

class OptimalControlProblem:

  def __init__(self, N, d, collocation_scheme, f, bounds):
    self.N = N
    self.d = d
    self.f = f
    self.num_states = f.size_in(0)[0]
    self.num_controls = f.size_in(1)[0]
    self.bounds = bounds

    self.C, self.D, self.B = self._calculate_collocation_coefficients(d, collocation_scheme)

  def _calculate_collocation_coefficients(self, d, collocation_scheme):
    tau_root = np.append(0.0, ca.collocation_points(d, collocation_scheme))

    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    B = np.zeros(d + 1)

    for j in range(d + 1):
      p = np.poly1d([1.0])
      for r in range(d + 1):
        if r != j:
          p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

      D[j] = p(1.0)

      pder = np.polyder(p)
      for r in range(d + 1):
        C[j, r] = pder(tau_root[r])

      pint = np.polyint(p)
      B[j] = pint(1.0)

    if collocation_scheme == 'radau':
      D = np.append(np.zeros(d), 1.0)

    return C, D, B

  def _initialize(self):
    self.w = {}
    self.w0 = []
    self.lbw = []
    self.ubw = []
    self.J = 0.0
    self.g = []
    self.lbg = []
    self.ubg = []

  def formulate(self):
    pass

  def _add_dynamics(self, h, initial_points):
    total_stage_cost = 0.0

    self._add_variable('x_0', self.num_states, 1, self.bounds['x_0']['lower'], self.bounds['x_0']['upper'], initial_points['x_0'])

    for k in range(self.N):
      self._add_variable(f'u_{k}', self.num_controls, 1, self.bounds[f'u_{k}']['lower'], self.bounds[f'u_{k}']['upper'], initial_points[f'u_{k}'])

      for j in range(self.d):
        self._add_variable(f'x_{k}_{j}', self.num_states, 1, self.bounds[f'x_{k}_{j}']['lower'], self.bounds[f'x_{k}_{j}']['upper'], initial_points[f'x_{k}_{j}'])

      xk_end = self.D[0] * self.w[f'x_{k}']
      for j in range(1, self.d + 1):
        xp = self.C[0, j] * self.w[f'x_{k}']
        for r in range(self.d): xp += self.C[r + 1, j] * self.w[f'x_{k}_{r}']

        fj, qj = self.f(self.w[f'x_{k}_{j - 1}'], self.w[f'u_{k}'])
        self._add_equality_constraint(h * fj - xp)

        xk_end += self.D[j] * self.w[f'x_{k}_{j - 1}']

        total_stage_cost += self.B[j] * qj * h

      self._add_variable(f'x_{k + 1}', self.num_states, 1, self.bounds[f'x_{k + 1}']['lower'], self.bounds[f'x_{k + 1}']['upper'], initial_points[f'x_{k + 1}'])

      self._add_equality_constraint(xk_end - self.w[f'x_{k + 1}'])

    return total_stage_cost

  def _add_variable(self, name, num_rows, num_columns, lower_bounds, upper_bounds, initial_point):
    variable = ca.MX.sym(name, num_rows, num_columns)
    self.w[name] = variable
    self.lbw.append(lower_bounds)
    self.ubw.append(upper_bounds)
    self.w0.append(initial_point)

  def _add_equality_constraint(self, expression):
    self.g.append(expression)
    self.lbg.append(np.zeros(expression.shape))
    self.ubg.append(np.zeros(expression.shape))

  def _add_inequality_constraint(self, expression, lower_bounds=None, upper_bounds=None):
    if lower_bounds is None or upper_bounds is None:
      lower_bounds = np.zeros(expression.shape)
      upper_bounds = np.full(expression.shape, np.inf)

    self.g.append(expression)
    self.lbg.append(lower_bounds)
    self.ubg.append(upper_bounds)

  def solve(self, solver_name, solver_options, save_path):
    w = ca.vertcat(*[ca.vec(variable) for variable in self.w.values()])
    g = ca.vertcat(*[ca.vec(expression) for expression in self.g])
    w0 = np.concatenate([value.flatten('F') for value in self.w0])
    lbw = np.concatenate([value.flatten('F') for value in self.lbw])
    ubw = np.concatenate([value.flatten('F') for value in self.ubw])
    lbg = np.concatenate([value.flatten('F') for value in self.lbg])
    ubg = np.concatenate([value.flatten('F') for value in self.ubg])

    prob = {'f': self.J, 'x': w, 'g': g}
    solver = ca.nlpsol('solver', solver_name, prob, solver_options)

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    split_indices = np.cumsum([v.numel() for v in self.w.values()])[:-1]

    split_values = np.split(sol['x'].full().flatten(), split_indices)

    reshaped_split_values = [split_values[i].reshape(v.shape, order='F') for i, v in enumerate(self.w.values())]

    solution = dict(zip(self.w.keys(), reshaped_split_values))

    with open(save_path, 'wb') as f:
      pickle.dump(solution, f)

class EnergyEfficiencyOptimalControlProblem(OptimalControlProblem):

  def __init__(self, N, d, collocation_scheme, f, bounds, body_weight, x_position_index, equal_state_index_map, opposite_state_index_map, gait_speed, f_vertical_ground_reaction_force):
    super().__init__(N, d, collocation_scheme, f, bounds)

    self.body_weight = body_weight
    self.x_position_index = x_position_index
    self.equal_state_indices_1 = list(equal_state_index_map.keys())
    self.equal_state_indices_2 = list(equal_state_index_map.values())
    self.opposite_state_indices_1 = list(opposite_state_index_map.keys())
    self.opposite_state_indices_2 = list(opposite_state_index_map.values())
    self.gait_speed = gait_speed
    self.f_vertical_ground_reaction_force = f_vertical_ground_reaction_force

  def formulate(self, initial_points):
    self._initialize()

    self._add_variable('tf', 1, 1, self.bounds['tf']['lower'], self.bounds['tf']['upper'], initial_points['tf'])
    h = self.w['tf'] / self.N

    metabolic_energy_consuption = self._add_dynamics(h, initial_points)

    travel_distance = self.w[f'x_{self.N}'][self.x_position_index] - self.w['x_0'][self.x_position_index]

    self.J = metabolic_energy_consuption / (self.body_weight * travel_distance)

    self._add_equality_constraint(self.w[f'x_{self.N}'][self.equal_state_indices_2] - self.w['x_0'][self.equal_state_indices_1])
    self._add_equality_constraint(self.w[f'x_{self.N}'][self.opposite_state_indices_2] + self.w['x_0'][self.opposite_state_indices_1])

    self._add_equality_constraint(self.gait_speed - travel_distance / self.w['tf'])

    self._add_inequality_constraint(self.f_vertical_ground_reaction_force(self.w['x_0']) - 0.1 * self.body_weight)
