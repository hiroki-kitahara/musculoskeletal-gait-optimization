import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.io
from scipy.signal import butter, filtfilt

SUBJECT_MASSES = {
  'AB01': 78.9,
  'AB02': 82.2,
  'AB03': 113.5,
  'AB05': 71.5,
  'AB06': 79.1,
  'AB07': 62.3,
  'AB08': 87.6,
  'AB09': 84.1,
  'AB10': 67.5,
  'AB11': 65.1,
  'AB12': 64.0,
  'AB13': 67.6,
}

def _process_emg_signal(raw_emg, fs, cutoff):
  emg_rectified = np.abs(raw_emg - np.mean(raw_emg, axis=0))

  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(4, normal_cutoff, btype='low', analog=False)

  emg_envelope = filtfilt(b, a, emg_rectified, axis=0)

  return emg_envelope

def _get_normalization_factors(subject_dir, subject_id, task_speed):
  ref_task_glob = list(subject_dir.glob(f'normal_walk_*_{task_speed}'))

  if not ref_task_glob:
    return None

  ref_task_folder = ref_task_glob[0]
  ref_task_name = ref_task_folder.name
  
  emg_file = ref_task_folder / f'{subject_id}_{ref_task_name}_emg.csv'

  df = pd.read_csv(emg_file)
  raw_data = df.iloc[:, 1:].values
  processed_data = _process_emg_signal(raw_data, fs=2000, cutoff=6.0)

  return np.max(processed_data, axis=0)

def _interpolate_cycle(data, num_points):
    t_original = np.linspace(0, 1, len(data))
    t_interp = np.linspace(0, 1, num_points)

    f = interp1d(t_original, data, axis=0, kind='linear')

    return f(t_interp)

def get_experimental_gait_data(experimental_data_dir, task_speed, task_basename='normal_walk', num_eval_points=101):
  processed_dir = experimental_data_dir / 'ProcessedData'
  segmentation_dir = experimental_data_dir / 'Segmentation'

  all_cycles = {
      'joint_angles': [],
      'joint_moments': [],
      'ground_reaction_forces': [],
      'muscle_activations': [],
      'cycle_duration': [],
      'stance_percentage': []
  }

  data_labels = {}

  subject_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('AB')])

  for subject_dir in subject_dirs:
    subject_id = subject_dir.name

    subject_mass = SUBJECT_MASSES[subject_id]

    max_emg_vals = _get_normalization_factors(subject_dir, subject_id, task_speed)

    task_folders = sorted(subject_dir.glob(f'{task_basename}_*_{task_speed}'))

    for task_folder in task_folders:
      task_name = task_folder.name

      segmentation_file = segmentation_dir / subject_id / f'{task_name}_parsing.mat'
      seg_data = scipy.io.loadmat(segmentation_file)
      cycles_times = seg_data['left']

      file_prefix = task_folder / f'{subject_id}_{task_name}'
      angle_df = pd.read_csv(f'{file_prefix}_angle.csv')
      moment_df = pd.read_csv(f'{file_prefix}_moment_filt.csv')
      grf_df = pd.read_csv(f'{file_prefix}_grf.csv')
      emg_df = pd.read_csv(f'{file_prefix}_emg.csv')

      time_200hz = angle_df.iloc[:, 0].values
      time_2000hz = emg_df.iloc[:, 0].values

      raw_emg = emg_df.iloc[:, 1:].values
      processed_emg = _process_emg_signal(raw_emg, fs=2000, cutoff=6.0)

      if max_emg_vals is not None:
        norm_denom = np.where(max_emg_vals == 0, 1.0, max_emg_vals)
        normalized_emg = processed_emg / norm_denom
      else:
        current_max = np.max(processed_emg, axis=0)
        norm_denom = np.where(current_max == 0, 1.0, current_max)
        normalized_emg = processed_emg / norm_denom

      emg_processed_df = pd.DataFrame(normalized_emg, columns=emg_df.columns[1:])

      data_map = {
        'joint_angles': (angle_df.iloc[:, 1:], time_200hz),
        'joint_moments': (moment_df.iloc[:, 1:], time_200hz),
        'ground_reaction_forces': (grf_df.iloc[:, 1:], time_200hz),
        'muscle_activations': (emg_processed_df, time_2000hz)
      }

      if not data_labels:
        for key, (df, _) in data_map.items():
          data_labels[key] = df.columns.tolist()

      for t_start, t_end in cycles_times:
        all_cycles['cycle_duration'].append(t_end - t_start)

        for key, (data_df, time_vec) in data_map.items():
          idx_start = np.argmin(np.abs(time_vec - t_start))
          idx_end = np.argmin(np.abs(time_vec - t_end))

          cycle_data_raw = data_df[data_labels[key]].values[idx_start:idx_end]

          cycle_data_interp = _interpolate_cycle(cycle_data_raw, num_eval_points)

          if key == 'joint_angles':
            cycle_data_interp = cycle_data_interp * np.pi / 180
          elif key == 'ground_reaction_forces':
            cycle_data_interp = cycle_data_interp / (subject_mass * 9.81) * 100

            count = 0
            for i in range(len(cycle_data_raw)):
              if cycle_data_raw[i, data_labels['ground_reaction_forces'].index('LForceY_Vertical')] > 0.1 * subject_mass * 9.81:
                count += 1
            all_cycles['stance_percentage'].append(count / len(cycle_data_raw))

          all_cycles[key].append(cycle_data_interp)

  gait_data_exp = {}
  gait_data_exp['cycle'] = np.linspace(0, 100, num_eval_points)
  for key, cycles_list in all_cycles.items():
    if key in ['cycle_duration', 'stance_percentage']:
      gait_data_exp[key] = np.array(cycles_list)
    else:
      gait_data_exp[key] = np.stack(cycles_list, axis=0)

  return gait_data_exp, data_labels
