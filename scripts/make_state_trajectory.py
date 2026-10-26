import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from scripts.build_musculoskeletal_system import build_musculoskeletal_system
from scripts.config import Config
from scripts.make_casadi_functions import make_f_grf_z
from scripts.process_cycle import construct_full_cycle, detect_touchdown_index, roll_x_full_cycle

def make_state_trajectory():
  parser = argparse.ArgumentParser()
  parser.add_argument('config_path', type=Path)
  parser.add_argument('solution_path', type=Path)
  parser.add_argument('save_path', type=Path)
  args = parser.parse_args()

  config = Config(args.config_path)

  with open(args.solution_path, 'rb') as f:
    solution = pickle.load(f)

  t_full_cycle, x_full_cycle = construct_full_cycle(solution, config.num_states, config.num_intervals, config.x_position_index, config.equal_state_index_map, config.opposite_state_index_map)

  musculoskeletal_system = build_musculoskeletal_system(config)
  f_grf_z = make_f_grf_z(musculoskeletal_system, config.num_muscles, config.num_coordinates, config.contact_names)

  touchdown_index = detect_touchdown_index(x_full_cycle, f_grf_z, config.body_weight)

  x_full_cycle_rolled = roll_x_full_cycle(x_full_cycle, touchdown_index, config.x_position_index)

  data = np.hstack([t_full_cycle[:, np.newaxis], x_full_cycle_rolled])
  columns = ['time'] + config.state_names

  state_trajectory = pd.DataFrame(data, columns=columns)
  state_trajectory.to_csv(args.save_path, index=False)

if __name__ == '__main__':
  make_state_trajectory()
