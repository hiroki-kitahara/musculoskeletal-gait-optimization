import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from msk.visualize import visualize

def take_snapshots():
  parser = argparse.ArgumentParser()
  parser.add_argument('state_trajectory_path', type=Path)
  parser.add_argument('opensim_model_path')
  parser.add_argument('save_dir', type=Path)
  parser.add_argument('-n', '--num-snapshots', default=11, type=int)
  args = parser.parse_args()

  state_trajectory = pd.read_csv(args.state_trajectory_path)

  t = state_trajectory['time'].to_numpy()
  x = state_trajectory.drop(columns=['time']).to_numpy()

  f = interp1d(t, x, axis=0)

  t_new = np.linspace(t[0], t[-1], args.num_snapshots)
  x_new = f(t_new)

  data_new = np.hstack([t_new[:, np.newaxis], x_new])

  state_trajectory_new = pd.DataFrame(data_new, columns=state_trajectory.columns)

  visualize(state_trajectory_new, args.opensim_model_path, args.save_dir)

if __name__ == '__main__':
  take_snapshots()
