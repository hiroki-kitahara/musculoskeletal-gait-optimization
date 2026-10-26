import casadi as ca
import numpy as np

from msk.contact import SmoothHuntCrossleyContact
from msk.ligament import AndersonPandyLigament
from msk.metabolic_energy import BhargavaMetabolicEnergy
from msk.muscle import DeGrooteRigidTendonMuscle
from msk.muscle_activation_dynamics import RaaschMuscleActivationDynamics
from msk.muscle_kinematics import PolynomialMuscleKinematics
from msk.musculoskeletal_system import MusculoskeletalSystem
from scripts.make_casadi_functions import make_f_px_mobilized_body

MUSCLE_DENSITY = 1059.7

def build_musculoskeletal_system(config):
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
    config.pennation_angles_at_optimal,
    config.max_contraction_velocities,
    config.fiber_dampings,
    config.constant_force_length_ratios,
    config.constant_force_velocity_ratios
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

  return MusculoskeletalSystem(
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

def build_musculoskeletal_system_on_uneven_terrain(config, step_x_position, step_width, step_height):
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
    config.pennation_angles_at_optimal,
    config.max_contraction_velocities,
    config.fiber_dampings
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

  musculoskeletal_system = build_musculoskeletal_system(config)

  f_px_mobilized_body = make_f_px_mobilized_body(musculoskeletal_system, config.num_muscles, config.num_coordinates)

  a = ca.SX.sym('a', config.num_muscles)
  q = ca.SX.sym('q', config.num_coordinates)
  v = ca.SX.sym('v', config.num_coordinates)

  x_list = [a, q, v]

  x = ca.vertcat(*x_list)

  px_mobilized_body = f_px_mobilized_body(x)

  contacts = []
  for i in range(config.num_contacts):
    pz = ca.if_else(px_mobilized_body[i] > step_x_position - step_width / 2, step_height, 0.0)
    pz = ca.if_else(px_mobilized_body[i] > step_x_position + step_width / 2, 0.0, pz)

    half_space_transform_with_step = ca.SX(np.array(config.half_space_transforms[i]))
    half_space_transform_with_step[2, 3] = pz

    contact = SmoothHuntCrossleyContact(
      config.sphere_positions_in_mobilized_body[i],
      config.sphere_radii[i],
      half_space_transform_with_step,
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

  return MusculoskeletalSystem(
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
  ), x_list
