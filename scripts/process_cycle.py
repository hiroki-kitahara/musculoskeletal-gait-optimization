import numpy as np

TOUCHDOWN_THRESHOLD_RATIO = 0.1

def construct_full_cycle(solution, num_states, num_intervals, x_position_index, equal_state_index_map, opposite_state_index_map):
  equal_state_indices_1 = list(equal_state_index_map.keys())
  equal_state_indices_2 = list(equal_state_index_map.values())
  opposite_state_indices_1 = list(opposite_state_index_map.keys())
  opposite_state_indices_2 = list(opposite_state_index_map.values())

  t_full_cycle = np.linspace(0.0, 2 * solution['tf'].item(), 2 * num_intervals + 1)

  x_first_half = np.empty((num_intervals + 1, num_states))
  for k in range(num_intervals + 1):
    x_first_half[k, :] = solution[f'x_{k}'].flatten()

  x_second_half = np.empty((num_intervals + 1, num_states))
  x_second_half[:, x_position_index] = x_first_half[:, x_position_index] + x_first_half[-1, x_position_index]
  x_second_half[:, equal_state_indices_2] = x_first_half[:, equal_state_indices_1]
  x_second_half[:, opposite_state_indices_2] = -x_first_half[:, opposite_state_indices_1]

  x_full_cycle = np.concatenate([x_first_half, x_second_half[1:]])

  return t_full_cycle, x_full_cycle

def detect_touchdown_index(x_full_cycle, f_grf_z, body_weight):
  threshold = TOUCHDOWN_THRESHOLD_RATIO * body_weight

  grf_prev = f_grf_z(x_full_cycle[-1]).full().item()

  for i, x in enumerate(x_full_cycle[:-1]):
    grf = f_grf_z(x).full().item()

    if grf_prev <= threshold < grf:
      return i

    grf_prev = grf

  return 0

def roll_x_full_cycle(x_full_cycle, touchdown_index, x_position_index):
  x_first_half = x_full_cycle[touchdown_index:, :].copy()
  x_first_half[:, x_position_index] -= x_first_half[0, x_position_index]
  x_second_half = x_full_cycle[: touchdown_index + 1, :].copy()
  x_second_half[:, x_position_index] += x_first_half[-1, x_position_index]
  return np.vstack([x_first_half, x_second_half[1:]])
