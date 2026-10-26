import argparse
from pathlib import Path
import pickle

import numpy as np

from scripts.build_musculoskeletal_system import build_musculoskeletal_system, build_musculoskeletal_system_on_uneven_terrain
from scripts.config import Config
from scripts.make_casadi_functions import make_f_grf_z, make_f_px_mobilized_body, make_f_sim
from scripts.process_cycle import construct_full_cycle, detect_touchdown_index, roll_x_full_cycle
from msk.simulation import Simulation

def run_simulation():
  parser = argparse.ArgumentParser()
  parser.add_argument('config_path', type=Path)
  parser.add_argument('solution_path', type=Path)
  parser.add_argument('save_path', type=Path)
  parser.add_argument('-m', '--monitoring-contact-name', default='heel_r')
  parser.add_argument('-w', '--step-width', default=0.5, type=float)
  parser.add_argument('-e', '--step-height', default=0.0, type=float)
  parser.add_argument('-p', '--perturbation-magnitude', default=0.0, type=float)
  parser.add_argument('-t', '--perturbation-timing-percent', default=0.0, type=float)
  parser.add_argument('-n', '--num-cycles', default=100, type=int)
  parser.add_argument('-z', '--z-position-threshold', default=0.7, type=float)
  args = parser.parse_args()

  config = Config(args.config_path)

  musculoskeletal_system = build_musculoskeletal_system(config)

  with open(args.solution_path, 'rb') as f:
    solution = pickle.load(f)

  t_full_cycle, x_full_cycle = construct_full_cycle(solution, config.num_states, config.num_intervals, config.x_position_index, config.equal_state_index_map, config.opposite_state_index_map)

  f_grf_z = make_f_grf_z(musculoskeletal_system, config.num_muscles, config.num_coordinates, config.contact_names)
  touchdown_index = detect_touchdown_index(x_full_cycle, f_grf_z, config.body_weight)

  if args.step_height != 0.0:
    x_full_cycle_rolled = roll_x_full_cycle(x_full_cycle, touchdown_index, config.x_position_index)

    f_px_mobilized_body = make_f_px_mobilized_body(musculoskeletal_system, config.num_muscles, config.num_coordinates)
    px_mobilized_body = f_px_mobilized_body(x_full_cycle_rolled[-1])[config.contact_names.index(args.monitoring_contact_name)]
    step_x_position = px_mobilized_body - x_full_cycle_rolled[0, config.x_position_index]
    print(f'Step x position: {step_x_position} m')

    musculoskeletal_system_on_uneven_terrain, x_list = build_musculoskeletal_system_on_uneven_terrain(config, step_x_position, args.step_width, args.step_height)
    f_sim = make_f_sim(musculoskeletal_system_on_uneven_terrain, config.num_muscles, config.num_coordinates, x_list=x_list)
  else:
    f_sim = make_f_sim(musculoskeletal_system, config.num_muscles, config.num_coordinates)

  if args.perturbation_magnitude != 0.0:
    perturbation_step = touchdown_index + np.argmin(np.abs(t_full_cycle - args.perturbation_timing_percent / 100 * 2 * solution['tf'].item()))
    h = solution['tf'].item() / config.num_intervals
    x_coordinate_index = config.coordinate_names.index(config.x_coordinate_name)

    def get_perturbation(k):
      perturbation = np.zeros((config.num_coordinates, 1))
      if k == perturbation_step:
        perturbation[x_coordinate_index, 0] = args.perturbation_magnitude / h
      return perturbation
  else:
    def get_perturbation(k):
      return np.zeros((config.num_coordinates, 1))

  simulation = Simulation(
    config.num_intervals,
    config.num_collocation_points,
    config.collocation_scheme,
    f_sim,
    config.x_position_index,
    config.equal_state_index_map,
    config.opposite_state_index_map,
    config.equal_control_index_map,
    config.opposite_control_index_map,
    solution,
    touchdown_index,
    args.num_cycles,
    config.z_position_index,
    args.z_position_threshold,
    config.state_names,
    args.save_path,
    get_perturbation=get_perturbation
  )

  return simulation.run()

if __name__ == '__main__':
  run_simulation()
