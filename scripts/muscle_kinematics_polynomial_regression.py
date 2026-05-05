import copy
import itertools
import json

import numpy as np
import opensim as osim

OPENSIM_MODEL_PATH = 'models/2d_human_musculoskeletal_model/opensim_model/gait10dof18musc.osim'

ACTUATED_COORDINATE_NAMES_PATH = 'models/2d_human_musculoskeletal_model/actuated_coordinate_names.json'
MUSCLE_NAMES_PATH = 'models/2d_human_musculoskeletal_model/muscle_names.json'
RANGE_OF_MOTION_PATH = 'models/2d_human_musculoskeletal_model/range_of_motion.json'

NUM_SAMPLES_DETECTION = 10
ABSOLUTE_MOMENT_ARM_THRESHOLD = 1e-4

NUM_SAMPLES_REGRESSION = 1000
INITIAL_DEGREE = 3
MAXIMUM_DEGREE = 9
ERROR_THRESHOLD = 3e-3

SAVE_PATH = 'models/2d_human_musculoskeletal_model/muscle_kinematic_parameters.json'

def main():
  model = osim.Model(OPENSIM_MODEL_PATH)
  state = model.initSystem()

  coordinate_set = model.getCoordinateSet()
  muscle_set = model.getMuscles()

  with open(ACTUATED_COORDINATE_NAMES_PATH, 'r') as f:
    actuated_coordinate_names = json.load(f)

  with open(RANGE_OF_MOTION_PATH, 'r') as f:
    range_of_motion = json.load(f)

  actuated_coordinates = []
  minimum_actuated_coordinate_values = []
  maximum_actuated_coordinate_values = []

  for coordinate_name in actuated_coordinate_names:
    coordinate = coordinate_set.get(coordinate_name)

    actuated_coordinates.append(coordinate)

    minimum_actuated_coordinate_values.append(range_of_motion[coordinate_name]['lower'])
    maximum_actuated_coordinate_values.append(range_of_motion[coordinate_name]['upper'])

  with open(MUSCLE_NAMES_PATH, 'r') as f:
    muscle_names = json.load(f)

  right_muscle_names = [muscle_name for muscle_name in muscle_names if muscle_name.endswith('_r')]
  right_muscles = [muscle_set.get(muscle_name) for muscle_name in right_muscle_names]

  sampled_actuated_coordinate_value_array = np.random.uniform(minimum_actuated_coordinate_values, maximum_actuated_coordinate_values, (NUM_SAMPLES_DETECTION, len(actuated_coordinate_names)))

  maximum_absolute_moment_arms = {muscle_name: {coordinate_name: 0.0 for coordinate_name in actuated_coordinate_names} for muscle_name in right_muscle_names}

  for actuated_coordinate_values in sampled_actuated_coordinate_value_array:
    for i, coordinate in enumerate(actuated_coordinates):
      coordinate.setValue(state, actuated_coordinate_values[i])

    model.realizePosition(state)

    for muscle_name, muscle in zip(right_muscle_names, right_muscles):
      for coordinate_name, coordinate in zip(actuated_coordinate_names, actuated_coordinates):
        moment_arm = muscle.computeMomentArm(state, coordinate)

        absolute_moment_arm = abs(moment_arm)
        if absolute_moment_arm > maximum_absolute_moment_arms[muscle_name][coordinate_name]:
          maximum_absolute_moment_arms[muscle_name][coordinate_name] = absolute_moment_arm

  right_muscle_spanned_coordinates_map = {}
  for muscle_name in right_muscle_names:
    spanned_coordinate_names = [coordinate_name for coordinate_name in actuated_coordinate_names if maximum_absolute_moment_arms[muscle_name][coordinate_name] > ABSOLUTE_MOMENT_ARM_THRESHOLD]
    right_muscle_spanned_coordinates_map[muscle_name] = spanned_coordinate_names

  right_muscle_kinematic_parameters = {}

  for muscle_name, spanned_coordinate_names in right_muscle_spanned_coordinates_map.items():
    print(f'\nProcessing muscle: {muscle_name}')

    spanned_coordinates = []
    minimum_spanned_coordinate_values = []
    maximum_spanned_coordinate_values = []

    for coordinate_name in spanned_coordinate_names:
      coordinate = coordinate_set.get(coordinate_name)

      spanned_coordinates.append(coordinate)

      minimum_spanned_coordinate_values.append(range_of_motion[coordinate_name]['lower'])
      maximum_spanned_coordinate_values.append(range_of_motion[coordinate_name]['upper'])

    muscle = muscle_set.get(muscle_name)

    sampled_spanned_coordinate_value_array = np.random.uniform(minimum_spanned_coordinate_values, maximum_spanned_coordinate_values, (NUM_SAMPLES_REGRESSION, len(spanned_coordinate_names)))

    sampled_muscle_lengths = []
    sampled_moment_arms = {coordinate_name: [] for coordinate_name in spanned_coordinate_names}

    for spanned_coordinate_values in sampled_spanned_coordinate_value_array:
      for i, coordinate in enumerate(spanned_coordinates):
        coordinate.setValue(state, spanned_coordinate_values[i])

      model.realizePosition(state)

      sampled_muscle_lengths.append(muscle.getLength(state))

      for coordinate_name, coordinate in zip(spanned_coordinate_names, spanned_coordinates):
        sampled_moment_arms[coordinate_name].append(muscle.computeMomentArm(state, coordinate))

    sampled_muscle_lengths = np.array(sampled_muscle_lengths)

    for coordinate_name in spanned_coordinate_names:
      sampled_moment_arms[coordinate_name] = np.array(sampled_moment_arms[coordinate_name])

    for degree in range(INITIAL_DEGREE, MAXIMUM_DEGREE + 1):
      exponents_list = []

      for combination in itertools.combinations_with_replacement(range(len(spanned_coordinate_names) + 1), degree):
        exponents = [0] * len(spanned_coordinate_names)
        for index in combination:
          if index:
            exponents[index - 1] += 1

        exponents_list.append(exponents)

      num_terms = len(exponents_list)

      A_lMT = np.zeros((NUM_SAMPLES_REGRESSION, num_terms))
      A_dMT = {coordinate_name: np.zeros((NUM_SAMPLES_REGRESSION, num_terms)) for coordinate_name in spanned_coordinate_names}

      for i, spanned_coordinate_values in enumerate(sampled_spanned_coordinate_value_array):
        for j, exponents in enumerate(exponents_list):
          value = np.prod(np.power(spanned_coordinate_values, exponents))
          A_lMT[i, j] = value

          for k, coordinate_name in enumerate(spanned_coordinate_names):
            if exponents[k] > 0:
              value = exponents[k] * np.power(spanned_coordinate_values[k], exponents[k] - 1)
              for m in range(len(spanned_coordinate_names)):
                if m != k:
                  value *= np.power(spanned_coordinate_values[m], exponents[m])

              A_dMT[coordinate_name][i, j] = -value

            else:
              A_dMT[coordinate_name][i, j] = 0.0

      A_list = [A_lMT]
      b_list = [sampled_muscle_lengths]

      for coordinate_name in spanned_coordinate_names:
        A_list.append(A_dMT[coordinate_name])
        b_list.append(sampled_moment_arms[coordinate_name])

      A = np.vstack(A_list)
      b = np.hstack(b_list)

      coefficients, *_ = np.linalg.lstsq(A, b, rcond=None)

      reconstructed_muscle_lengths = A_lMT @ coefficients
      muscle_length_error = np.sqrt(np.mean((sampled_muscle_lengths - reconstructed_muscle_lengths)**2))

      moment_arm_error_list = []

      for coordinate_name in spanned_coordinate_names:
        reconstructed_moment_arms = A_dMT[coordinate_name] @ coefficients
        moment_arm_error = np.sqrt(np.mean((sampled_moment_arms[coordinate_name] - reconstructed_moment_arms)**2))

        moment_arm_error_list.append(moment_arm_error)

      maximum_moment_arm_error = max(moment_arm_error_list)

      if muscle_length_error < ERROR_THRESHOLD and maximum_moment_arm_error < ERROR_THRESHOLD:
        break

    print(f'  Degree: {degree} | RMSE of muscle lengths: {muscle_length_error:.5f} m | RMSE of moment arms: {maximum_moment_arm_error:.5f} m')

    right_muscle_kinematic_parameters[muscle_name] = {'spanned_coordinate_names': spanned_coordinate_names, 'exponents_list': exponents_list, 'coefficients': coefficients.tolist()}

  left_muscle_kinematic_parameters = {}
  for muscle_name, parameters in right_muscle_kinematic_parameters.items():
    left_muscle_name = muscle_name[:-2] + '_l'
    left_muscle_kinematic_parameters[left_muscle_name] = copy.deepcopy(parameters)
    left_muscle_kinematic_parameters[left_muscle_name]['spanned_coordinate_names'] = [coordinate_name[:-2] + '_l' for coordinate_name in parameters['spanned_coordinate_names']]

  muscle_kinematic_parameters = right_muscle_kinematic_parameters | left_muscle_kinematic_parameters

  with open(SAVE_PATH, 'w') as f:
    json.dump(muscle_kinematic_parameters, f, indent=2)

if __name__ == '__main__':
  main()
