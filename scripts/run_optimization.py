import argparse
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from msk.optimal_control_problem import EnergyEfficiencyOptimalControlProblem, StabilityOptimalControlProblem
from scripts.build_musculoskeletal_system import build_musculoskeletal_system
from scripts.config import Config
from scripts.make_casadi_functions import make_f, make_f_grf_z, make_A

def _randomize_initial_guess(initial_guess, bounds, noise_scale, rng):
  randomized_initial_guess = {}

  for variable_name, value in initial_guess.items():
    lower_bounds = bounds[variable_name]['lower']
    upper_bounds = bounds[variable_name]['upper']

    difference = upper_bounds - lower_bounds
    difference[np.isinf(difference)] = 1.0

    noise_range = difference * noise_scale
    noise_value = rng.uniform(-noise_range / 2.0, noise_range / 2.0)

    randomized_value = value + noise_value
    randomized_initial_guess[variable_name] = randomized_value

  return randomized_initial_guess

def run_optimization():
  parser = argparse.ArgumentParser()
  parser.add_argument('task')
  parser.add_argument('config_path', type=Path)
  args = parser.parse_args()

  config = Config(args.config_path, args.task)

  musculoskeletal_system = build_musculoskeletal_system(config)

  f = make_f(musculoskeletal_system, config.num_muscles, config.num_coordinates)
  f_grf_z = make_f_grf_z(musculoskeletal_system, config.num_muscles, config.num_coordinates, config.contact_names)

  if args.task == 'energy_efficiency_optimization':
    optimal_control_problem = EnergyEfficiencyOptimalControlProblem(
      config.num_intervals,
      config.num_collocation_points,
      config.collocation_scheme,
      f,
      config.bounds,
      config.body_mass * config.gravitational_acceleration,
      config.x_position_index,
      config.equal_state_index_map,
      config.opposite_state_index_map,
      config.gait_speed,
      f_grf_z
    )

  elif args.task == 'stability_optimization':
    A = make_A(musculoskeletal_system, config.num_muscles, config.num_coordinates)

    optimal_control_problem = StabilityOptimalControlProblem(
      config.num_intervals,
      config.num_collocation_points,
      config.collocation_scheme,
      f,
      config.bounds,
      config.body_mass * config.gravitational_acceleration,
      config.x_position_index,
      config.equal_state_index_map,
      config.opposite_state_index_map,
      config.gait_speed,
      f_grf_z,
      A,
      config.sensitivity_mask,
      config.equal_control_index_map,
      config.opposite_control_index_map,
      config.cost_of_transport
    )

  rng = np.random.default_rng(config.seed)
  config.save_directory.mkdir(parents=True, exist_ok=True)

  for i in range(config.num_iterations):
    randomized_initial_guess = _randomize_initial_guess(config.initial_guess, config.bounds, config.noise_scale, rng)

    optimal_control_problem.formulate(randomized_initial_guess)

    with open(config.save_directory / f'log{i + 1}.txt', 'w') as f:
      with redirect_stdout(f):
        optimal_control_problem.solve(config.solver_name, config.solver_options, config.save_directory / f'solution{i + 1}.pkl')

if __name__ == '__main__':
  run_optimization()
