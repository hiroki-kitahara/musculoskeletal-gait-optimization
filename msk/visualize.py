import math
from pathlib import Path
import time

import cv2
import opensim as osim

def _create_lookat_transform(camera_pos, target_pos, up_dir=osim.Vec3(0, 1, 0)):
  z_axis = osim.Vec3(
    camera_pos.get(0) - target_pos.get(0),
    camera_pos.get(1) - target_pos.get(1),
    camera_pos.get(2) - target_pos.get(2)
  )
  
  norm_z = math.sqrt(z_axis.get(0)**2 + z_axis.get(1)**2 + z_axis.get(2)**2)
  if norm_z < 1e-9:
    return osim.Transform()
  z_axis = osim.Vec3(z_axis.get(0) / norm_z, z_axis.get(1) / norm_z, z_axis.get(2) / norm_z)

  x_axis = osim.Vec3(
    up_dir.get(1) * z_axis.get(2) - up_dir.get(2) * z_axis.get(1),
    up_dir.get(2) * z_axis.get(0) - up_dir.get(0) * z_axis.get(2),
    up_dir.get(0) * z_axis.get(1) - up_dir.get(1) * z_axis.get(0)
  )
  
  norm_x = math.sqrt(x_axis.get(0)**2 + x_axis.get(1)**2 + x_axis.get(2)**2)
  if norm_x < 1e-9:
    return osim.Transform()
  x_axis = osim.Vec3(x_axis.get(0) / norm_x, x_axis.get(1) / norm_x, x_axis.get(2) / norm_x)

  y_axis = osim.Vec3(
    z_axis.get(1) * x_axis.get(2) - z_axis.get(2) * x_axis.get(1),
    z_axis.get(2) * x_axis.get(0) - z_axis.get(0) * x_axis.get(2),
    z_axis.get(0) * x_axis.get(1) - z_axis.get(1) * x_axis.get(0)
  )

  rot = osim.Rotation()
  rot.set(0, 0, x_axis.get(0)); rot.set(0, 1, y_axis.get(0)); rot.set(0, 2, z_axis.get(0))
  rot.set(1, 0, x_axis.get(1)); rot.set(1, 1, y_axis.get(1)); rot.set(1, 2, z_axis.get(1))
  rot.set(2, 0, x_axis.get(2)); rot.set(2, 1, y_axis.get(2)); rot.set(2, 2, z_axis.get(2))

  return osim.Transform(rot, camera_pos)

def visualize(
    state_trajectory,
    opensim_model_path,
    save_directory_path,
    show_ground_plane=False,
    show_step=False,
    step_position=0.0,
    step_width=0.5,
    step_height=0.01,
    background_color=(1.0, 1.0, 1.0),
    automatic_camera_orbit=True,
    x_position_name='pelvis_tx_position',
    camera_height=0.9,
    camera_distance=3.0,
    camera_speed=0.0
):
  model = osim.Model(opensim_model_path)

  if show_ground_plane:
    tile_size = 0.5
    num_x = 40
    num_z = 40
    thickness = 0.01
    height_offset = 0.03

    color_even = osim.Vec3(0.8, 0.8, 0.8)
    color_odd = osim.Vec3(0.5, 0.5, 0.5)

    start_x = - (num_x * tile_size) / 2.0
    start_z = - (num_z * tile_size) / 2.0

    ground = model.getGround()

    for ix in range(num_x):
      for iz in range(num_z):
        half_dims = osim.Vec3(tile_size/2, thickness/2, tile_size/2)
        brick = osim.Brick(half_dims)

        if (ix + iz) % 2 == 0:
          brick.upd_Appearance().set_color(color_even)
        else:
          brick.upd_Appearance().set_color(color_odd)

        pos_x = start_x + ix * tile_size
        pos_z = start_z + iz * tile_size
        pos_y = -thickness / 2.0 - height_offset

        location = osim.Vec3(pos_x, pos_y, pos_z)
        transform = osim.Transform(location)

        frame_name = f'TileFrame_{ix}_{iz}'
        tile_frame = osim.PhysicalOffsetFrame(frame_name, ground, transform)

        tile_frame.attachGeometry(brick)

        model.addComponent(tile_frame)

  if show_step:
    half_dims = osim.Vec3(step_width / 2, step_height / 2, 0.5)
    brick = osim.Brick(half_dims)

    color = osim.Vec3(1, 1, 0)
    brick.upd_Appearance().set_color(color)

    height_offset = 0.03
    pos_x = step_position
    pos_z = 0
    pos_y = step_height / 2 - height_offset

    location = osim.Vec3(pos_x, pos_y, pos_z)
    transform = osim.Transform(location)

    frame_name = 'Step'
    step_frame = osim.PhysicalOffsetFrame(frame_name, ground, transform)

    step_frame.attachGeometry(brick)

    model.addComponent(step_frame)

  model.setUseVisualizer(True)

  state = model.initSystem()

  opensim_visualizer = model.getVisualizer()
  simbody_visualizer = opensim_visualizer.getSimbodyVisualizer()

  simbody_visualizer.setBackgroundType(simbody_visualizer.SolidColor)
  simbody_visualizer.setBackgroundColor(osim.Vec3(*background_color))

  opensim_visualizer.show(state)

  input('Maximize the OpenSim Visualizer, click the "View" button in the lower-left corner, and select "Save Movie". Then return to this terminal and press Enter:')

  for i in range(len(state_trajectory)):
    for col in state_trajectory.columns:
      if col.endswith('_position'):
        coordinate_name = col[:-len('_position')]
        coordinate_set = model.getCoordinateSet()
        if coordinate_set.contains(coordinate_name):
          coordinate = coordinate_set.get(coordinate_name)

          position = state_trajectory.at[i, col]

          coordinate.setValue(state, position)

      elif col.endswith('_activation'):
        muscle_name = col[:-len('_activation')]
        muscle_set = model.getMuscles()
        if muscle_set.contains(muscle_name):
          muscle = muscle_set.get(muscle_name)

          activation = state_trajectory.at[i, col]

          muscle.setActivation(state, activation)

    model.realizeDynamics(state)

    if automatic_camera_orbit:
      camera_position = osim.Vec3(state_trajectory.at[i, x_position_name], camera_height, camera_distance)
      target_position = osim.Vec3(state_trajectory.at[i, x_position_name], camera_height, 0.0)
    else:
      camera_position = osim.Vec3(camera_speed * state_trajectory.at[i, 'time'], camera_height, camera_distance)
      target_position = osim.Vec3(camera_speed * state_trajectory.at[i, 'time'], camera_height, 0.0)

    camera_transform = _create_lookat_transform(camera_position, target_position)

    simbody_visualizer.setCameraTransform(camera_transform)

    opensim_visualizer.show(state)

    watch_directory = Path('Python_1')
    target_file = watch_directory / f'Frame{i+1:04d}.png'
    while not target_file.exists():
      time.sleep(0.01)

  watch_directory.rename(save_directory_path)

def create_video_from_images(input_folder, output_file, fps=30):
  images = sorted([f for f in input_folder.iterdir() if f.is_file() and f.name.endswith('.png')])

  frame = cv2.imread(str(images[0]))
  height, width, layers = frame.shape

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  video = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

  for image_path in images:
    img = cv2.imread(str(image_path))
    video.write(img)

  video.release()
  cv2.destroyAllWindows()
