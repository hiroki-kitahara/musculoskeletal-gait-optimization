import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from msk.visualize import visualize, create_video_from_images

def take_movie():
  parser = argparse.ArgumentParser()
  parser.add_argument('state_trajectory_path', type=Path)
  parser.add_argument('opensim_model_path')
  parser.add_argument('save_dir', type=Path)
  parser.add_argument('-f', '--fps', default=30.0, type=float)
  parser.add_argument('-s', '--playback-speed', default=1.0, type=float)
  parser.add_argument('-p', '--step-x-position', default=0.0, type=float)
  parser.add_argument('-w', '--step-width', default=0.5, type=float)
  parser.add_argument('-e', '--step-height', default=0.0, type=float)
  parser.add_argument('-v', '--camera-speed', default=0.0, type=float)
  args = parser.parse_args()

  state_trajectory = pd.read_csv(args.state_trajectory_path)

  t = state_trajectory['time'].to_numpy()
  x = state_trajectory.drop(columns=['time']).to_numpy()

  f = interp1d(t, x, axis=0)

  dt_sim = args.playback_speed / args.fps
  t_new = np.arange(t[0], t[-1], dt_sim)
  x_new = f(t_new)

  data_new = np.hstack([t_new[:, np.newaxis], x_new])

  state_trajectory_new = pd.DataFrame(data_new, columns=state_trajectory.columns)

  show_step = args.step_height != 0.0
  automatic_camera_orbit = args.camera_speed == 0.0

  visualize(
    state_trajectory_new,
    args.opensim_model_path,
    args.save_dir,
    show_ground_plane=True,
    show_step=show_step,
    step_position=args.step_x_position,
    step_width=args.step_width,
    step_height=args.step_height,
    background_color=(0.0, 0.0, 0.0),
    automatic_camera_orbit=automatic_camera_orbit,
    camera_speed=args.camera_speed
  )

  create_video_from_images(args.save_dir, args.save_dir / 'movie.mp4', args.fps)

if __name__ == '__main__':
  take_movie()
