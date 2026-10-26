import json
from pathlib import Path
import pickle
import platform
from typing import Any

import casadi as ca
import pandas as pd
import numpy as np
import numpy.typing as npt

class Config:
  coordinate_names: list[str]
  actuated_coordinate_names: list[str]
  muscle_names: list[str]
  contact_names: list[str]
  state_names: list[str]
  control_names: list[str]

  num_coordinates: int
  num_actuated_coordinates: int
  num_muscles: int
  num_contacts: int
  num_states: int
  num_controls: int

  spanned_coordinate_indices_list: list[list[int]]
  exponents_lists: list[list[list[float]]]
  coefficients_list: list[list[float]]

  max_isometric_forces: ca.DM
  optimal_fiber_lengths: ca.DM
  tendon_slack_lengths: ca.DM
  pennation_angles_at_optimal: ca.DM
  max_contraction_velocities: ca.DM
  fiber_dampings: ca.DM
  constant_force_length_ratios: ca.DM
  constant_force_velocity_ratios: ca.DM

  k0: ca.DM
  k1: ca.DM
  k2: ca.DM
  k3: ca.DM
  k4: ca.DM
  theta: ca.DM
  phi: ca.DM

  mobilized_body_names: list[str]
  sphere_positions_in_mobilized_body: list[ca.DM]
  sphere_radii: list[float]
  half_space_transforms: list[ca.DM]
  stiffnesses: list[float]
  dissipations: list[float]
  static_frictions: list[float]
  dynamic_frictions: list[float]
  viscous_frictions: list[float]
  transition_velocities: list[float]
  constant_contact_forces: list[float]
  hertz_smoothings: list[float]
  hunt_crossley_smoothings: list[float]

  specific_tensions: ca.DM
  slow_twitch_fiber_ratios: ca.DM

  body_mass: float
  gravitational_acceleration: float
  body_weight: float
  damping: float
  x_coordinate_name: str
  z_coordinate_name: str
  x_position_index: int
  z_position_index: int

  contact_kinematics_functions: list[ca.Function]
  skeletal_dynamics_function: ca.Function

  num_intervals: int
  num_collocation_points: int
  collocation_scheme: str

  gait_speed: float
  cost_of_transport: float | None

  equal_state_index_map: dict[int, int]
  opposite_state_index_map: dict[int, int]
  equal_control_index_map: dict[int, int]
  opposite_control_index_map: dict[int, int]

  sensitivity_mask: npt.NDArray[np.float64] | None

  bounds: dict[str, npt.NDArray[np.float64]]
  initial_guess: dict[str, npt.NDArray[np.float64]]

  num_iterations: int
  seed: int
  noise_scale: float
  solver_name: str
  solver_options: dict[str, Any]
  save_directory: Path

  def __init__(self, config_path, task=None):
    system_name = platform.system()

    config = self._load_json(config_path)

    model_dir = Path(config['model_dir'])
    optimization_config_dir = Path(config['optimization_config_dir'])

    self.coordinate_names = self._load_json(model_dir / 'coordinate_names.json')
    self.actuated_coordinate_names = self._load_json(model_dir / 'actuated_coordinate_names.json')
    self.muscle_names = self._load_json(model_dir / 'muscle_names.json')
    self.contact_names = self._load_json(model_dir / 'contact_names.json')
    self.state_names = [f'{muscle_name}_activation' for muscle_name in self.muscle_names] + [f'{coordinate_name}_position' for coordinate_name in self.coordinate_names] + [f'{coordinate_name}_velocity' for coordinate_name in self.coordinate_names]
    self.control_names = [f'{muscle_name}_excitation' for muscle_name in self.muscle_names]

    self.num_coordinates = len(self.coordinate_names)
    self.num_actuated_coordinates = len(self.actuated_coordinate_names)
    self.num_muscles = len(self.muscle_names)
    self.num_contacts = len(self.contact_names)
    self.num_states = len(self.state_names)
    self.num_controls = len(self.control_names)

    self._load_muscle_kinematic_parameters(model_dir / 'muscle_kinematic_parameters.json')
    self._load_muscle_parameters(model_dir / 'muscle_parameters.csv')
    self._load_ligament_parameters(model_dir / 'ligament_parameters.csv')
    self._load_contact_parameters(model_dir / 'contact_parameters.json')
    self._load_specific_tensions(model_dir / 'specific_tensions.csv')
    self._load_slow_twitch_fiber_ratios(model_dir / 'slow_twitch_fiber_ratios.csv')
    self._load_musculoskeletal_system_parameters(model_dir / 'musculoskeletal_system_parameters.json')

    self.contact_kinematics_functions = [self._load_external_function(model_dir, f'f_{contact_name}_kinematics', system_name) for contact_name in self.contact_names]
    self.skeletal_dynamics_function = self._load_external_function(model_dir, 'f_skeletal_dynamics', system_name)

    self._apply_overrides(self.constant_force_length_ratios, config.get('constant_force_length_ratios'), self.muscle_names)
    self._apply_overrides(self.constant_force_velocity_ratios, config.get('constant_force_velocity_ratios'), self.muscle_names)
    self._apply_overrides(self.stiffnesses, config.get('stiffnesses'), self.contact_names)

    self.num_intervals = config['num_intervals']
    self.num_collocation_points = config['num_collocation_points']
    self.collocation_scheme = config['collocation_scheme']

    self.gait_speed = config['gait_speed']
    self.cost_of_transport = config['cost_of_transport'] if task == 'stability_optimization' else None

    self.equal_state_index_map, self.opposite_state_index_map, self.equal_control_index_map, self.opposite_control_index_map = self._make_symmetric_variable_index_maps(optimization_config_dir / 'symmetric_variable_name_maps.json')

    self.sensitivity_mask = self._make_sensitivity_mask() if task == 'stability_optimization' else None

    self.bounds = self._make_bounds(optimization_config_dir / 'bounds.json', task)
    self.initial_guess = self._make_initial_guess(optimization_config_dir / 'initial_guess.json', task, config.get('half_gait_cycle_duration_guess'))

    if config.get('initial_guess_path'):
      with open(config['initial_guess_path'], 'rb') as f:
        self.initial_guess.update(pickle.load(f))

    self.num_iterations = config['num_iterations']
    self.seed = config['seed']
    self.noise_scale = config['noise_scale']
    self.solver_name = config['solver_name']
    self.solver_options = config['solver_options']
    self.save_directory = Path(config['save_dir'])

  def _load_muscle_kinematic_parameters(self, file_path):
    parameters = self._load_json(file_path)

    self.spanned_coordinate_indices_list = []
    self.exponents_lists = []
    self.coefficients_list = []

    for muscle_name in self.muscle_names:
      spanned_coordinate_names = parameters[muscle_name]['spanned_coordinate_names']
      spanned_coordinate_indices = [self.actuated_coordinate_names.index(name) for name in spanned_coordinate_names]
      self.spanned_coordinate_indices_list.append(spanned_coordinate_indices)

      self.exponents_lists.append(parameters[muscle_name]['exponents_list'])

      self.coefficients_list.append(parameters[muscle_name]['coefficients'])

  def _load_muscle_parameters(self, file_path):
    parameters = pd.read_csv(file_path, index_col=0)
    parameters = parameters.reindex(self.muscle_names)

    self.max_isometric_forces = ca.DM(parameters['max_isometric_force'])
    self.optimal_fiber_lengths = ca.DM(parameters['optimal_fiber_length'])
    self.tendon_slack_lengths = ca.DM(parameters['tendon_slack_length'])
    self.pennation_angles_at_optimal = ca.DM(parameters['pennation_angle_at_optimal'])
    self.max_contraction_velocities = ca.DM(parameters['max_contraction_velocity'])
    self.fiber_dampings = ca.DM(parameters['fiber_damping'])
    self.constant_force_length_ratios = ca.DM(parameters['constant_force_length_ratio'])
    self.constant_force_velocity_ratios = ca.DM(parameters['constant_force_velocity_ratio'])

  def _load_ligament_parameters(self, file_path):
    parameters = pd.read_csv(file_path, index_col=0)
    parameters = parameters.reindex(self.actuated_coordinate_names)

    self.k0 = ca.DM(parameters['k0'])
    self.k1 = ca.DM(parameters['k1'])
    self.k2 = ca.DM(parameters['k2'])
    self.k3 = ca.DM(parameters['k3'])
    self.k4 = ca.DM(parameters['k4'])
    self.theta = ca.DM(parameters['theta'])
    self.phi = ca.DM(parameters['phi'])

  def _load_contact_parameters(self, file_path):
    parameters = self._load_json(file_path)

    self.mobilized_body_names = []
    self.sphere_positions_in_mobilized_body = []
    self.sphere_radii = []
    self.half_space_transforms = []
    self.stiffnesses = []
    self.dissipations = []
    self.static_frictions = []
    self.dynamic_frictions = []
    self.viscous_frictions = []
    self.transition_velocities = []
    self.constant_contact_forces = []
    self.hertz_smoothings = []
    self.hunt_crossley_smoothings = []

    for contact_name in self.contact_names:
      self.mobilized_body_names.append(parameters[contact_name]['mobilized_body_name'])
      self.sphere_positions_in_mobilized_body.append(ca.DM(parameters[contact_name]['sphere_position_in_mobilized_body']))
      self.sphere_radii.append(parameters[contact_name]['sphere_radius'])
      self.half_space_transforms.append(ca.DM(parameters[contact_name]['half_space_transform']))
      self.stiffnesses.append(parameters[contact_name]['stiffness'])
      self.dissipations.append(parameters[contact_name]['dissipation'])
      self.static_frictions.append(parameters[contact_name]['static_friction'])
      self.dynamic_frictions.append(parameters[contact_name]['dynamic_friction'])
      self.viscous_frictions.append(parameters[contact_name]['viscous_friction'])
      self.transition_velocities.append(parameters[contact_name]['transition_velocity'])
      self.constant_contact_forces.append(parameters[contact_name]['constant_contact_force'])
      self.hertz_smoothings.append(parameters[contact_name]['hertz_smoothing'])
      self.hunt_crossley_smoothings.append(parameters[contact_name]['hunt_crossley_smoothing'])

  def _load_specific_tensions(self, file_path):
    parameters = pd.read_csv(file_path, index_col=0)
    parameters = parameters.reindex(self.muscle_names)

    self.specific_tensions = ca.DM(parameters['specific_tension'])

  def _load_slow_twitch_fiber_ratios(self, file_path):
    parameters = pd.read_csv(file_path, index_col=0)
    parameters = parameters.reindex(self.muscle_names)

    self.slow_twitch_fiber_ratios = ca.DM(parameters['slow_twitch_fiber_ratio'])

  def _load_musculoskeletal_system_parameters(self, file_path):
    parameters = self._load_json(file_path)

    self.body_mass = parameters['body_mass']
    self.gravitational_acceleration = parameters['gravitational_acceleration']
    self.body_weight = self.body_mass * self.gravitational_acceleration
    self.damping = parameters['damping']
    self.x_coordinate_name = parameters['x_coordinate_name']
    self.z_coordinate_name = parameters['z_coordinate_name']
    self.x_position_index = self.state_names.index(f'{parameters["x_coordinate_name"]}_position')
    self.z_position_index = self.state_names.index(f'{parameters["z_coordinate_name"]}_position')

  def _make_symmetric_variable_index_maps(self, file_path):
    name_maps = self._load_json(file_path)

    equal_state_index_map = {self.state_names.index(k): self.state_names.index(v) for k, v in name_maps['state']['equal'].items()}
    opposite_state_index_map = {self.state_names.index(k): self.state_names.index(v) for k, v in name_maps['state']['opposite'].items()}
    equal_control_index_map = {self.control_names.index(k): self.control_names.index(v) for k, v in name_maps['control']['equal'].items()}
    opposite_control_index_map = {self.control_names.index(k): self.control_names.index(v) for k, v in name_maps['control']['opposite'].items()}

    return equal_state_index_map, opposite_state_index_map, equal_control_index_map, opposite_control_index_map

  def _make_sensitivity_mask(self):
    sensitivity_mask = np.zeros((self.num_states, self.num_states))
    sensitivity_mask[:self.num_muscles, :self.num_muscles] = np.identity(self.num_muscles)
    sensitivity_mask[self.num_muscles:, self.num_muscles:] = 1
    return sensitivity_mask

  def _make_bounds(self, file_path, task):
    bounds_information = self._load_json(file_path)

    tf_lower_bound = np.array([[bounds_information['half_gait_cycle_duration']['lower']]])
    tf_upper_bound = np.array([[bounds_information['half_gait_cycle_duration']['upper']]])
    x_lower_bounds = np.array([[bounds_information[state_name]['lower'] for state_name in self.state_names]]).T
    x_upper_bounds = np.array([[bounds_information[state_name]['upper'] for state_name in self.state_names]]).T
    x0_lower_bounds = x_lower_bounds.copy()
    x0_lower_bounds[self.x_position_index, 0] = 0.0
    x0_upper_bounds = x_upper_bounds.copy()
    x0_upper_bounds[self.x_position_index, 0] = 0.0
    u_lower_bounds = np.array([[bounds_information[control_name]['lower'] for control_name in self.control_names]]).T
    u_upper_bounds = np.array([[bounds_information[control_name]['upper'] for control_name in self.control_names]]).T

    bounds = {}

    bounds['tf'] = {'lower': tf_lower_bound, 'upper': tf_upper_bound}
    bounds['x_0'] = {'lower': x0_lower_bounds, 'upper': x0_upper_bounds}
    for k in range(self.num_intervals):
      bounds[f'u_{k}'] = {'lower': u_lower_bounds, 'upper': u_upper_bounds}
      for j in range(self.num_collocation_points):
        bounds[f'x_{k}_{j}'] = {'lower': x_lower_bounds, 'upper': x_upper_bounds}
      bounds[f'x_{k + 1}'] = {'lower': x_lower_bounds, 'upper': x_upper_bounds}

    if task == 'stability_optimization':
      num_sensitivity_elements = np.count_nonzero(self.sensitivity_mask)
      selected_sensitivity_indices = np.where(self.sensitivity_mask.flatten('F'))[0]
      X_el_lower_bounds = np.full((num_sensitivity_elements, 1), -np.inf)
      X_el_upper_bounds = np.full((num_sensitivity_elements, 1), np.inf)
      identity_elements = np.identity(self.num_states).reshape((-1, 1), order='F')[selected_sensitivity_indices]

      bounds['X_el_0'] = {'lower': identity_elements, 'upper': identity_elements}
      for k in range(2 * self.num_intervals):
        for j in range(self.num_collocation_points):
          bounds[f'X_el_{k}_{j}'] = {'lower': X_el_lower_bounds, 'upper': X_el_upper_bounds}
        bounds[f'X_el_{k + 1}'] = {'lower': X_el_lower_bounds, 'upper': X_el_upper_bounds}

    return bounds

  def _make_initial_guess(self, file_path, task, half_gait_cycle_duration_guess):
    initial_guess_information = self._load_json(file_path)

    tf_initial_guess = initial_guess_information['half_gait_cycle_duration'] if half_gait_cycle_duration_guess is None else half_gait_cycle_duration_guess
    x_initial_guess = np.array([[initial_guess_information[state_name] for state_name in self.state_names]]).T
    u_initial_guess = np.array([[initial_guess_information[control_name] for control_name in self.control_names]]).T
    x_initial_guess_trajectory = np.tile(x_initial_guess, (self.num_intervals + 1, 1, 1))
    x_initial_guess_trajectory[:, self.x_position_index, 0] = np.linspace(0.0, self.gait_speed * tf_initial_guess, self.num_intervals + 1)

    initial_guess = {}

    initial_guess['tf'] = np.array([[tf_initial_guess]])
    initial_guess['x_0'] = x_initial_guess_trajectory[0]
    for k in range(self.num_intervals):
      initial_guess[f'u_{k}'] = u_initial_guess
      for j in range(self.num_collocation_points):
        initial_guess[f'x_{k}_{j}'] = x_initial_guess_trajectory[k]
      initial_guess[f'x_{k + 1}'] = x_initial_guess_trajectory[k + 1]

    if task == 'stability_optimization':
      selected_sensitivity_indices = np.where(self.sensitivity_mask.flatten('F'))[0]
      identity_elements = np.identity(self.num_states).reshape((-1, 1), order='F')[selected_sensitivity_indices]

      initial_guess['X_el_0'] = identity_elements
      for k in range(2 * self.num_intervals):
        for j in range(self.num_collocation_points):
          initial_guess[f'X_el_{k}_{j}'] = identity_elements
        initial_guess[f'X_el_{k + 1}'] = identity_elements

    return initial_guess

  def _load_json(self, file_path):
    with open(file_path, 'r') as f:
      return json.load(f)

  def _load_external_function(self, directory_path, base_name, system_name):
    file_name = f'{base_name}.dll' if system_name == 'Windows' else f'lib{base_name}.so'
    file_path = directory_path / file_name
    return ca.external(base_name, str(file_path))

  def _apply_overrides(self, target_array, override_values, labels):
    if isinstance(override_values, float):
      if isinstance(target_array, list):
        target_array[:] = [override_values] * len(target_array)
      else:
        target_array[:] = override_values

    elif isinstance(override_values, dict):
      for k, v in override_values.items():
        target_array[labels.index(k)] = v
