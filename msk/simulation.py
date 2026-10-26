import casadi as ca
import numpy as np
import pandas as pd
import scipy.optimize

from msk.optimal_control_problem import calculate_collocation_coefficients

class Simulation:

  def __init__(self, N, d, collocation_scheme, f, x_position_index, equal_state_index_map, opposite_state_index_map, equal_control_index_map, opposite_control_index_map, solution, k0, num_cycles, z_position_index, z_position_threshold, state_names, save_path, get_perturbation):
    self.N = N
    self.d = d
    self.C, self.D, self.B = calculate_collocation_coefficients(d, collocation_scheme)
    self.f = f
    self.num_states = f.size_in(0)[0]
    self.num_controls = f.size_in(1)[0]
    self.x_position_index = x_position_index
    self.equal_state_indices_1 = list(equal_state_index_map.keys())
    self.equal_state_indices_2 = list(equal_state_index_map.values())
    self.opposite_state_indices_1 = list(opposite_state_index_map.keys())
    self.opposite_state_indices_2 = list(opposite_state_index_map.values())
    self.equal_control_indices_1 = list(equal_control_index_map.keys())
    self.equal_control_indices_2 = list(equal_control_index_map.values())
    self.opposite_control_indices_1 = list(opposite_control_index_map.keys())
    self.opposite_control_indices_2 = list(opposite_control_index_map.values())
    self.solution = solution
    self.h = solution['tf'].item() / N
    self.k0 = k0
    self.num_cycles = num_cycles
    self.z_position_index = z_position_index
    self.z_position_threshold = z_position_threshold
    self.state_names = state_names
    self.save_path = save_path
    self.get_perturbation = get_perturbation

  def run(self):
    xk = self.solution[f'x_{self.k0 % self.N}'].copy()
    xk[self.x_position_index] = 0.0
    if self.k0 >= self.N:
      xk[self.equal_state_indices_2] = xk[self.equal_state_indices_1]
      xk[self.opposite_state_indices_2] = -xk[self.opposite_state_indices_1]

    t = [0.0]
    x = [xk.flatten()]
    status = 'success'

    for k in range(self.k0, self.num_cycles * 2 * self.N + self.k0):
      if (k - self.k0) % (2 * self.N) == 0:
        print(f'Cycle count: {k // (2 * self.N) + 1}')

      xk = self.solve_single_step_dynamics(k, xk)

      if xk is not None:
        t.append((k - self.k0 + 1) * self.h)
        x.append(xk.flatten())

      else:
        print('Error')
        status = 'error'
        break

      if xk[self.z_position_index, 0] < self.z_position_threshold:
        print('Fall down')
        status = 'fall_down'
        break

    else:
      print('Success')

    t = np.stack(t)
    x = np.stack(x)

    data = np.hstack([t[:, np.newaxis], x])
    columns = ['time'] + self.state_names

    state_trajectory = pd.DataFrame(data=data, columns=columns)
    state_trajectory.to_csv(self.save_path, index=False)

    return status

  def solve_single_step_dynamics(self, k, xk):
    x = {}
    g = []

    uk = self.solution[f'u_{k % self.N}'].copy()
    if (k // self.N) % 2:
      uk[self.equal_control_indices_2] = uk[self.equal_control_indices_1]
      uk[self.opposite_control_indices_2] = -uk[self.opposite_control_indices_1]

    dk = self.get_perturbation(k)

    for j in range(self.d):
      x[f'x_{k}_{j}'] = ca.MX.sym(f'x_{k}_{j}', self.num_states)

    xk_end = self.D[0] * xk

    for j in range(1, self.d + 1):
      xp = self.C[0, j] * xk
      for r in range(self.d): xp += self.C[r + 1, j] * x[f'x_{k}_{r}']

      xj = x[f'x_{k}_{j - 1}']
      fj = self.f(xj, uk, dk)
      g.append(self.h * fj - xp)

      xk_end += self.D[j] * xj

    xk_next = ca.MX.sym(f'x_{k + 1}', self.num_states)
    x[f'x_{k + 1}'] = xk_next

    g.append(xk_end - xk_next)

    x = ca.vertcat(*x.values())
    g = ca.vertcat(*g)

    x0 = np.tile(xk, (self.d + 1, 1))

    func_residual = ca.Function('res', [x], [g])

    def scipy_fun(x_in):
      res = func_residual(x_in.reshape(x0.shape))
      return np.array(res).flatten()

    sol_scipy = scipy.optimize.root(
      scipy_fun,
      x0.flatten(),
      method='hybr',
      options={'xtol': 1e-9, 'maxfev': 10000}
    )

    if sol_scipy.success:
      result_reshaped = sol_scipy.x.reshape(x0.shape)
      return result_reshaped[self.d * self.num_states :]
