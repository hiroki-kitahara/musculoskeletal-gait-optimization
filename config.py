import json
from pathlib import Path
import platform
from typing import Any

import casadi as ca
import pandas as pd
import numpy as np
import numpy.typing as npt

class Config:
  system_name: str

  model_directory: Path
  optimization_config_directory: Path

  coordinate_names: list[str]
  actuated_coordinate_names: list[str]
  muscle_names: list[str]
  contact_names: list[str]

  num_coordinates: int
  num_actuated_coordinates: int
  num_muscles: int
  num_contacts: int

  spanned_coordinate_indices_list: list[list[int]]
  exponents_lists: list[list[list[float]]]
  coefficients_list: list[list[float]]

  max_isometric_forces: ca.DM
  optimal_fiber_lengths: ca.DM
  tendon_slack_lengths: ca.DM
  pennation_angle_at_optimal: ca.DM
  max_contraction_velocities: ca.DM
  constant_force_length_ratio: ca.DM
  constant_force_velocity_ratio: ca.DM
  beta: ca.DM

  k0: ca.DM
  k1: ca.DM
  k2: ca.DM
  k3: ca.DM
  k4: ca.DM
  theta: ca.DM
  phi: ca.DM

  contact_kinematics_functions: list[ca.Function]

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

  skeletal_dynamics_function: ca.Function

  specific_tensions: ca.DM
  slow_twitch_fiber_ratios: ca.DM

  B: float

  state_names: list[str]
  control_names: list[str]

  num_states: int
  num_controls: int

  num_intervals: int
  num_collocation_points: int
  collocation_scheme: str

  body_weight: float
  x_position_index: int
  equal_state_index_map: dict[int, int]
  opposite_state_index_map: dict[int, int]
  gait_speed: float

  bounds: dict[str, npt.NDArray[np.float64]]
  initial_guess: dict[str, npt.NDArray[np.float64]]

  num_iterations: int
  seed: int
  noise_scale: float
  solver_name: str
  solver_options: dict[str, Any]
  save_directory: Path

  def __init__(self):
    self.system_name = platform.system()

    self.model_directory = Path('models/2d_human_musculoskeletal_model')
    self.optimization_config_directory = Path('optimization_config/2d_human_musculoskeletal_model')

    self.coordinate_names = self._load_json(self.model_directory / 'coordinate_names.json')
    self.actuated_coordinate_names = self._load_json(self.model_directory / 'actuated_coordinate_names.json')
    self.muscle_names = self._load_json(self.model_directory / 'muscle_names.json')
    self.contact_names = self._load_json(self.model_directory / 'contact_names.json')

    self.num_coordinates = len(self.coordinate_names)
    self.num_actuated_coordinates = len(self.actuated_coordinate_names)
    self.num_muscles = len(self.muscle_names)
    self.num_contacts = len(self.contact_names)

    self._load_muscle_kinematic_parameters(self.model_directory / 'muscle_kinematic_parameters.json')

    self._load_muscle_parameters(self.model_directory / 'muscle_parameters.csv')
    self.fiber_damping = ca.DM.ones(len(self.muscle_names)) * 0.01
    self.constant_force_length_ratio = ca.DM.zeros(len(self.muscle_names))
    self.constant_force_velocity_ratio = ca.DM.zeros(len(self.muscle_names))

    self._load_ligament_parameters(self.model_directory / 'ligament_parameters.csv')

    self.contact_kinematics_functions = [self._load_external_function(self.model_directory, f'f_{contact_name}_kinematics') for contact_name in self.contact_names]
    self._load_contact_parameters(self.model_directory / 'contact_parameters.json')

    self.skeletal_dynamics_function = self._load_external_function(self.model_directory, 'f_skeletal_dynamics')

    self.specific_tensions = self._load_csv_as_dm(self.model_directory / 'specific_tensions.csv', self.muscle_names, 'specific_tension')
    self.slow_twitch_fiber_ratios = self._load_csv_as_dm(self.model_directory / 'slow_twitch_fiber_ratios.csv', self.muscle_names, 'slow_twitch_fiber_ratio')

    self.damping = 0.1

    self.state_names = [f'{muscle_name}_activation' for muscle_name in self.muscle_names] + [f'{coordinate_name}_position' for coordinate_name in self.coordinate_names] + [f'{coordinate_name}_velocity' for coordinate_name in self.coordinate_names]
    self.control_names = [f'{muscle_name}_excitation' for muscle_name in self.muscle_names]

    self.num_states = len(self.state_names)
    self.num_controls = len(self.control_names)

    self.num_intervals = 30
    self.num_collocation_points = 3
    self.collocation_scheme = 'radau'

    self.body_weight = 7.37e2
    self.x_position_index = self.state_names.index('pelvis_tx_position')
    self.equal_state_index_map, self.opposite_state_index_map = self._generate_symmetric_variable_name_maps(self.optimization_config_directory / 'symmetric_variable_name_maps.json')
    self.gait_speed = 1.3

    self.bounds = self._generate_bounds(self.optimization_config_directory / 'bounds.json')
    self.initial_guess = self._generate_initial_guess_from_scratch(self.optimization_config_directory / 'initial_guess.json', 0.6)

    self.num_iterations = 5
    self.seed = 1234
    self.noise_scale = 1e-2
    self.solver_name = 'ipopt'
    self.solver_options = {
      'ipopt': {
        'hessian_approximation': 'limited-memory'
      }
    }
    self.save_directory = Path('results/2d_human_musculoskeletal_model')

  def _load_muscle_kinematic_parameters(self, path):
    parameters = self._load_json(path)

    self.spanned_coordinate_indices_list = []
    self.exponents_lists = []
    self.coefficients_list = []

    for muscle_name in self.muscle_names:
      spanned_coordinate_names = parameters[muscle_name]['spanned_coordinate_names']
      spanned_coordinate_indices = [self.actuated_coordinate_names.index(name) for name in spanned_coordinate_names]
      self.spanned_coordinate_indices_list.append(spanned_coordinate_indices)

      self.exponents_lists.append(parameters[muscle_name]['exponents_list'])

      self.coefficients_list.append(parameters[muscle_name]['coefficients'])

  def _load_muscle_parameters(self, path):
    parameters = pd.read_csv(path, index_col=0)
    parameters = parameters.reindex(self.muscle_names)

    self.max_isometric_forces = ca.DM(parameters['max_isometric_force'])
    self.optimal_fiber_lengths = ca.DM(parameters['optimal_fiber_length'])
    self.tendon_slack_lengths = ca.DM(parameters['tendon_slack_length'])
    self.pennation_angle_at_optimal = ca.DM(parameters['pennation_angle_at_optimal'])
    self.max_contraction_velocities = ca.DM(parameters['max_contraction_velocity'])

  def _load_ligament_parameters(self, path):
    parameters = pd.read_csv(path, index_col=0)
    parameters = parameters.reindex(self.actuated_coordinate_names)

    self.k0 = ca.DM(parameters['k0'])
    self.k1 = ca.DM(parameters['k1'])
    self.k2 = ca.DM(parameters['k2'])
    self.k3 = ca.DM(parameters['k3'])
    self.k4 = ca.DM(parameters['k4'])
    self.theta = ca.DM(parameters['theta'])
    self.phi = ca.DM(parameters['phi'])

  def _load_contact_parameters(self, path):
    parameters = self._load_json(path)

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

  def _generate_symmetric_variable_name_maps(self, path):
    symmetric_variable_name_maps = self._load_json(path)
    equal_state_index_map = {self.state_names.index(k): self.state_names.index(v) for k, v in symmetric_variable_name_maps['state']['equal'].items()}
    opposite_state_index_map = {self.state_names.index(k): self.state_names.index(v) for k, v in symmetric_variable_name_maps['state']['opposite'].items()}
    return equal_state_index_map, opposite_state_index_map

  def _generate_bounds(self, path):
    bounds_information = self._load_json(path)

    state_lower_bounds = np.array([[bounds_information[state_name]['lower'] for state_name in self.state_names]]).T
    state_upper_bounds = np.array([[bounds_information[state_name]['upper'] for state_name in self.state_names]]).T
    control_lower_bounds = np.array([[bounds_information[control_name]['lower'] for control_name in self.control_names]]).T
    control_upper_bounds = np.array([[bounds_information[control_name]['upper'] for control_name in self.control_names]]).T

    bounds = {
      'tf': {
        'lower': np.array([[bounds_information['half_gait_cycle_duration']['lower']]]),
        'upper': np.array([[bounds_information['half_gait_cycle_duration']['upper']]])
      },
      'x_0': {
        'lower': state_lower_bounds.copy(),
        'upper': state_upper_bounds.copy()
      }
    }

    bounds['x_0']['lower'][self.x_position_index, 0] = 0.0
    bounds['x_0']['upper'][self.x_position_index, 0] = 0.0

    for k in range(self.num_intervals):
      bounds[f'u_{k}'] = {'lower': control_lower_bounds, 'upper': control_upper_bounds}

      for j in range(self.num_collocation_points):
        bounds[f'x_{k}_{j}'] = {'lower': state_lower_bounds, 'upper': state_upper_bounds}

      bounds[f'x_{k + 1}'] = {'lower': state_lower_bounds, 'upper': state_upper_bounds}

    return bounds

  def _generate_initial_guess_from_scratch(self, path, half_gait_cycle_duration):
    initial_guess_information = self._load_json(path)

    state_initial_guess = np.array([[initial_guess_information[state_name] for state_name in self.state_names]]).T
    control_initial_guess = np.array([[initial_guess_information[control_name] for control_name in self.control_names]]).T

    state_initial_guess_trajectory = np.tile(state_initial_guess, (self.num_intervals + 1, 1, 1))
    control_initial_guess_trajectory = np.tile(control_initial_guess, (self.num_intervals, 1, 1))

    state_initial_guess_trajectory[:, self.x_position_index, 0] = np.linspace(0.0, self.gait_speed * half_gait_cycle_duration, self.num_intervals + 1)

    initial_guess = {
      'tf': np.array([[half_gait_cycle_duration]]),
      'x_0': state_initial_guess_trajectory[0]
    }

    for k in range(self.num_intervals):
      initial_guess[f'u_{k}'] = control_initial_guess_trajectory[k]

      for j in range(self.num_collocation_points):
        initial_guess[f'x_{k}_{j}'] = state_initial_guess_trajectory[k]

      initial_guess[f'x_{k + 1}'] = state_initial_guess_trajectory[k + 1]

    return initial_guess

  def _load_json(self, path):
    with open(path, 'r') as f:
      return json.load(f)

  def _load_external_function(self, directory_path, base_name):
    file_name = f'{base_name}.dll' if self.system_name == 'Windows' else f'lib{base_name}.so'
    file_path = directory_path / file_name
    return ca.external(base_name, str(file_path))

  def _load_csv_as_dm(self, path, index_names, target_column):
    df = pd.read_csv(path, index_col=0)
    df = df.reindex(index_names)
    return ca.DM(df[target_column])
