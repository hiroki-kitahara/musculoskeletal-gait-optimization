import sys

import casadi as ca
import numpy as np

from config import Config
from src.contact import SmoothHuntCrossleyContact
from src.ligament import AndersonPandyLigament
from src.metabolic_energy import BhargavaMetabolicEnergy
from src.muscle import DeGrooteRigidTendonMuscle
from src.muscle_activation_dynamics import RaaschMuscleActivationDynamics
from src.muscle_kinematics import PolynomialMuscleKinematics
from src.musculoskeletal_system import MusculoskeletalSystem
from src.optimal_control_problem import EnergyEfficiencyOptimalControlProblem

MUSCLE_DENSITY = 1059.7

def energy_efficiency_optimization(config: Config):
  actuated_coordinate_indices = [config.coordinate_names.index(coordinate_name) for coordinate_name in config.actuated_coordinate_names]

  muscle_activation_dynamics = RaaschMuscleActivationDynamics()

  muscle_kinematics = PolynomialMuscleKinematics(
    config.spanned_coordinate_indices_list,
    config.exponents_lists,
    config.coefficients_list
  )

  muscle = DeGrooteRigidTendonMuscle(
    config.max_isometric_forces,
    config.optimal_fiber_lengths,
    config.tendon_slack_lengths,
    config.pennation_angle_at_optimal,
    config.max_contraction_velocities,
    config.fiber_damping
  )

  ligament = AndersonPandyLigament(
    config.k0,
    config.k1,
    config.k2,
    config.k3,
    config.k4,
    config.theta,
    config.phi
  )

  contacts = []
  for i in range(config.num_contacts):
    contact = SmoothHuntCrossleyContact(
      config.sphere_positions_in_mobilized_body[i],
      config.sphere_radii[i],
      config.half_space_transforms[i],
      config.stiffnesses[i],
      config.dissipations[i],
      config.static_frictions[i],
      config.dynamic_frictions[i],
      config.viscous_frictions[i],
      config.transition_velocities[i],
      config.constant_contact_forces[i],
      config.hertz_smoothings[i],
      config.hunt_crossley_smoothings[i]
    )
    contacts.append(contact)

  muscle_masses = MUSCLE_DENSITY * config.max_isometric_forces / config.specific_tensions * config.optimal_fiber_lengths
  metabolic_energy = BhargavaMetabolicEnergy(
    muscle_masses,
    config.slow_twitch_fiber_ratios
  )

  musculoskeletal_system = MusculoskeletalSystem(
    actuated_coordinate_indices,
    muscle_activation_dynamics,
    muscle_kinematics,
    muscle,
    ligament,
    config.contact_kinematics_functions,
    contacts,
    config.skeletal_dynamics_function,
    metabolic_energy,
    config.damping
  )

  a = ca.SX.sym('a', config.num_muscles)
  q = ca.SX.sym('q', config.num_coordinates)
  v = ca.SX.sym('v', config.num_coordinates)

  x = [a, q, v]

  u = ca.SX.sym('u', config.num_muscles)

  x_dot = musculoskeletal_system.get_state_derivatives(x, u)

  x = ca.vertcat(*x)
  x_dot = ca.vertcat(*x_dot)
  L = ca.sum(musculoskeletal_system.E_dot)

  f = ca.Function('f', [x, u], [x_dot, L], ['x', 'u'], ['x_dot', 'L'])

  grf_z = sum(W_mobilized_body[2] for W_mobilized_body in musculoskeletal_system.W_mobilized_body_list)

  f_vertical_ground_reaction_force = ca.Function('f_vertical_ground_reaction_force', [x], [grf_z], ['x'], ['grf_z'])

  energy_efficiency_optimal_control_problem = EnergyEfficiencyOptimalControlProblem(
    config.num_intervals,
    config.num_collocation_points,
    config.collocation_scheme,
    f,
    config.bounds,
    config.body_weight,
    config.x_position_index,
    config.equal_state_index_map,
    config.opposite_state_index_map,
    config.gait_speed,
    f_vertical_ground_reaction_force
  )

  rng = np.random.default_rng(config.seed)
  config.save_directory.mkdir(parents=True, exist_ok=True)

  for i in range(config.num_iterations):
    randomized_initial_guess = {}

    for variable_name, value in config.initial_guess.items():
      lower_bounds = config.bounds[variable_name]['lower']
      upper_bounds = config.bounds[variable_name]['upper']

      difference = upper_bounds - lower_bounds
      difference[np.isinf(difference)] = 1.0

      noise_range = difference * config.noise_scale
      noise_value = rng.uniform(-noise_range / 2.0, noise_range / 2.0)

      randomized_value = value + noise_value
      randomized_initial_guess[variable_name] = randomized_value

    energy_efficiency_optimal_control_problem.formulate(randomized_initial_guess)

    with open(config.save_directory / f'log{i + 1}.txt', 'w') as f:
      sys.stdout = f
      energy_efficiency_optimal_control_problem.solve(config.solver_name, config.solver_options, config.save_directory / f'solution{i + 1}.pkl')

if __name__ == '__main__':
  config = Config()
  energy_efficiency_optimization(config)
