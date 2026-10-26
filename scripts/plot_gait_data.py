import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.build_musculoskeletal_system import build_musculoskeletal_system
from scripts.config import Config
from scripts.make_casadi_functions import make_f_gait_data
from msk.get_experimental_gait_data import get_experimental_gait_data

TOUCHDOWN_THRESHOLD_RATIO = 0.1
CM_TO_INCH = 1 / 2.54

XCORR_FONTSIZE = 7.5
XCORR_LINE_SPACING = 1.5
XCORR_MARGIN_PT = 3.0

def _get_simulation_gait_data(state_trajectory_path, config_path):
  state_trajectory = pd.read_csv(state_trajectory_path)
  t_full_cycle = state_trajectory['time'].to_numpy()
  x_full_cycle = state_trajectory.drop(columns=['time']).to_numpy()

  config = Config(config_path)
  musculoskeletal_system = build_musculoskeletal_system(config)
  f_gait_data = make_f_gait_data(musculoskeletal_system, config.num_muscles, config.num_coordinates, config.contact_names)

  position_indices = [config.state_names.index(f'{name}_position') for name in config.actuated_coordinate_names]
  activation_indices = [config.state_names.index(f'{name}_activation') for name in config.muscle_names]

  joint_angles = []
  joint_moments = []
  ground_reaction_forces = []
  muscle_activations = []
  for x in x_full_cycle:
    tau, grf = f_gait_data(x)
    joint_angles.append(x[position_indices])
    joint_moments.append(tau.full().flatten() / config.body_mass)
    ground_reaction_forces.append(grf.full().flatten() / config.body_weight * 100)
    muscle_activations.append(x[activation_indices])

  joint_angles = np.stack(joint_angles)
  joint_moments = np.stack(joint_moments)
  ground_reaction_forces = np.stack(ground_reaction_forces)
  muscle_activations = np.stack(muscle_activations)

  cycle = np.linspace(0, 100, len(t_full_cycle))
  cycle_duration = t_full_cycle[-1] - t_full_cycle[0]
  stance_percentage = np.count_nonzero(ground_reaction_forces[:-1, 2] > TOUCHDOWN_THRESHOLD_RATIO * 100) / (len(t_full_cycle) - 1)

  data_sim = {
    'cycle': cycle,
    'joint_angles': joint_angles,
    'joint_moments': joint_moments,
    'ground_reaction_forces': ground_reaction_forces,
    'muscle_activations': muscle_activations,
    'cycle_duration': cycle_duration,
    'stance_percentage': stance_percentage,
  }
  columns_sim = {
    'joint_angles': config.actuated_coordinate_names,
    'joint_moments': config.actuated_coordinate_names,
    'ground_reaction_forces': ['x', 'y', 'z'],
    'muscle_activations': config.muscle_names,
  }
  return data_sim, columns_sim

def _plot_sim_curves(ax, data_sim_list, columns_sim_list, key, name_sim, scale_sim, colors_sim, labels_sim, linestyles_sim, linewidth_sim):
  for data_sim, columns_sim, color_sim, label_sim, linestyle_sim in zip(data_sim_list, columns_sim_list, colors_sim, labels_sim, linestyles_sim):
    index_sim = columns_sim[key].index(name_sim)
    ax.plot(data_sim['cycle'], data_sim[key][:, index_sim] * scale_sim, color=color_sim, linestyle=linestyle_sim, linewidth=linewidth_sim, label=label_sim)

def _plot_exp_band(ax, cycle_exp, values_exp, color_exp, alpha_exp, label_exp):
  mean_exp = np.nanmean(values_exp, axis=0)
  std_exp = np.nanstd(values_exp, axis=0)
  ax.fill_between(cycle_exp, mean_exp - std_exp, mean_exp + std_exp, color=color_exp, alpha=alpha_exp, label=label_exp, zorder=0)

def _plot_xcorr(ax, data_sim_list, columns_sim_list, key, name_sim, scale_sim, cycle_exp, values_exp, colors_sim, loc='best', fontsize=XCORR_FONTSIZE):
  texts = []
  colors = []

  mean_exp = np.nanmean(values_exp, axis=0)
  for data_sim, columns_sim, color_sim in zip(data_sim_list, columns_sim_list, colors_sim):
    index_sim = columns_sim[key].index(name_sim)
    sim_cycle = data_sim['cycle']
    sim_val = data_sim[key][:, index_sim] * scale_sim
    sim_val_interp = np.interp(cycle_exp, sim_cycle, sim_val)

    sim_norm = sim_val_interp - np.mean(sim_val_interp)
    exp_norm = mean_exp - np.mean(mean_exp)
    std_sim = np.sqrt(np.sum(sim_norm**2))
    std_exp = np.sqrt(np.sum(exp_norm**2))

    if std_sim > 1e-6 and std_exp > 1e-6:
      corr = np.correlate(sim_norm, exp_norm, mode='full') / (std_sim * std_exp)
      lags = np.arange(-len(cycle_exp) + 1, len(cycle_exp))
      max_idx = np.argmax(corr)
      max_corr = corr[max_idx]
      max_lag = lags[max_idx]
      lag_pct = max_lag / (len(cycle_exp) - 1) * 100
      text = rf'$r = {max_corr:.2f}$, $\Delta t = {math.floor(lag_pct)} \, \%$'
    else:
      text = 'R=NaN, lag=NaN'

    texts.append(text)
    colors.append(color_sim)

  handles = [Line2D([], [], linestyle='none') for _ in texts]
  legend = Legend(
    ax, handles, texts,
    loc=loc,
    frameon=False,
    handlelength=0,
    handletextpad=0,
    borderpad=0.2,
    borderaxespad=0.3,
    labelspacing=0.15,
    fontsize=fontsize,
    labelcolor=colors,
  )
  ax.add_artist(legend)
  return legend

def _data_occupancy(ax, y_lower, y_upper):
  count = 0
  for line in ax.get_lines():
    y_values = np.asarray(line.get_ydata(), dtype=float)
    if y_values.size == 0:
      continue
    count += int(np.count_nonzero((y_values >= y_lower) & (y_values <= y_upper)))
  for collection in ax.collections:
    for path in collection.get_paths():
      y_values = np.asarray(path.vertices[:, 1], dtype=float)
      if y_values.size == 0:
        continue
      count += int(np.count_nonzero((y_values >= y_lower) & (y_values <= y_upper)))
  return count

def _add_headroom(ax, n_lines, fontsize=XCORR_FONTSIZE, line_spacing=XCORR_LINE_SPACING, margin_pt=XCORR_MARGIN_PT):
  if n_lines <= 0:
    return

  height_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
  if height_pt <= 0:
    return

  required_pt = line_spacing * n_lines * fontsize + margin_pt
  fraction = required_pt / height_pt
  if fraction >= 1.0:
    return

  y_min, y_max = ax.get_ylim()
  span = y_max - y_min
  if span <= 0:
    return

  band = fraction * span
  occupancy_upper = _data_occupancy(ax, y_max - band, y_max)
  occupancy_lower = _data_occupancy(ax, y_min, y_min + band)

  ticks = [tick for tick in ax.get_yticks() if y_min <= tick <= y_max]

  if occupancy_upper <= occupancy_lower:
    ax.set_ylim(y_min, y_max + band)
  else:
    ax.set_ylim(y_min - band, y_max)

  if ticks:
    ax.set_yticks(ticks)

def _add_headroom_for_xcorr(fig):
  fig.draw_without_rendering()
  for ax in fig.axes:
    n_lines = sum(len(artist.get_texts()) for artist in ax.get_children() if isinstance(artist, Legend))
    _add_headroom(ax, n_lines)

def plot_gait_data():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--state-trajectory-paths', nargs='+', type=Path, required=True)
  parser.add_argument('-c', '--config-paths', nargs='+', type=Path, required=True)
  parser.add_argument('-o', '--save-path', required=True, type=Path)
  parser.add_argument('-C', '--colors-sim', nargs='+')
  parser.add_argument('-l', '--labels-sim', nargs='+')
  parser.add_argument('-L', '--linestyles-sim', nargs='+')
  parser.add_argument('-e', '--no-experimental', dest='show_experimental', action='store_false')
  parser.add_argument('-d', '--experimental-data-dir', default=Path('experimental_data'), type=Path)
  parser.add_argument('-t', '--task-speed', default='1-2')
  parser.add_argument('-x', '--show-xcorr', action='store_true')
  args = parser.parse_args()

  data_sim_list = []
  columns_sim_list = []
  for state_trajectory_path, config_path in zip(args.state_trajectory_paths, args.config_paths):
    data_sim, columns_sim = _get_simulation_gait_data(state_trajectory_path, config_path)
    data_sim_list.append(data_sim)
    columns_sim_list.append(columns_sim)

  show_experimental = args.show_experimental
  data_exp = None
  columns_exp = None
  if show_experimental:
    data_exp, columns_exp = get_experimental_gait_data(args.experimental_data_dir, args.task_speed)

  colors_sim = args.colors_sim if args.colors_sim else plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(data_sim_list)]
  linestyles_sim = args.linestyles_sim if args.linestyles_sim else ['-'] * len(data_sim_list)
  linewidth_sim = 1.0
  labels_sim = args.labels_sim if args.labels_sim else [f'Sim {i+1}' for i in range(len(args.state_trajectory_paths))]

  color_exp = '#bfbfbf'
  alpha_exp = 1.0
  capsize_exp = 1.5
  label_exp = 'Exp'

  params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],

    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,

    'mathtext.fontset': 'stix',

    'pdf.fonttype': 42
  }
  plt.rcParams.update(params)

  fig, axes = plt.subplots(5, 3, sharex=True, figsize=(0.9 * 16.9998 * CM_TO_INCH, 1.40625 * 0.9 * 16.9998 * CM_TO_INCH), layout='constrained')

  joint_titles = ['Hip', 'Knee', 'Ankle']
  joint_names_sim = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
  angle_names_exp = ['hip_flexion_l', 'knee_angle_l', 'ankle_angle_l']
  for i in range(3):
    ax = axes[0, i]
    ax.set_title(joint_titles[i])
    _plot_sim_curves(ax, data_sim_list, columns_sim_list, 'joint_angles', joint_names_sim[i], 180 / np.pi, colors_sim, labels_sim, linestyles_sim, linewidth_sim)
    if show_experimental:
      index_exp = columns_exp['joint_angles'].index(angle_names_exp[i])
      values_exp = data_exp['joint_angles'][:, :, index_exp] * 180 / np.pi
      _plot_exp_band(ax, data_exp['cycle'], values_exp, color_exp, alpha_exp, label_exp)
      if args.show_xcorr:
        _plot_xcorr(ax, data_sim_list, columns_sim_list, 'joint_angles', joint_names_sim[i], 180 / np.pi, data_exp['cycle'], values_exp, colors_sim)
    ax.set_xlim(0, 100)
    if i == 0:
      ax.set_ylabel(r'Angle ($\text{degree}$)')

  moment_names_exp = ['hip_flexion_l_moment', 'knee_angle_l_moment', 'ankle_angle_l_moment']
  for i in range(3):
    ax = axes[1, i]
    ax.set_title(joint_titles[i])
    _plot_sim_curves(ax, data_sim_list, columns_sim_list, 'joint_moments', joint_names_sim[i], 1.0, colors_sim, labels_sim, linestyles_sim, linewidth_sim)
    if show_experimental:
      index_exp = columns_exp['joint_moments'].index(moment_names_exp[i])
      values_exp = data_exp['joint_moments'][:, :, index_exp]
      _plot_exp_band(ax, data_exp['cycle'], values_exp, color_exp, alpha_exp, label_exp)
      if args.show_xcorr:
        _plot_xcorr(ax, data_sim_list, columns_sim_list, 'joint_moments', joint_names_sim[i], 1.0, data_exp['cycle'], values_exp, colors_sim)
    ax.set_xlim(0, 100)
    if i == 0:
      ax.set_ylabel(r'Moment ($\text{N} \cdot \text{m} / \text{kg}$)')

  grf_titles = ['Fore-Aft', 'Vertical']
  grf_names_sim = ['x', 'z']
  grf_names_exp = ['LForceZ', 'LForceY_Vertical']
  grf_flip_flags = [True, False]
  for i in range(2):
    ax = axes[2, i]
    ax.set_title(grf_titles[i])
    _plot_sim_curves(ax, data_sim_list, columns_sim_list, 'ground_reaction_forces', grf_names_sim[i], 1.0, colors_sim, labels_sim, linestyles_sim, linewidth_sim)
    if show_experimental:
      index_exp = columns_exp['ground_reaction_forces'].index(grf_names_exp[i])
      values_exp = data_exp['ground_reaction_forces'][:, :, index_exp]
      if grf_flip_flags[i]:
        values_exp = -values_exp
      _plot_exp_band(ax, data_exp['cycle'], values_exp, color_exp, alpha_exp, label_exp)
      if args.show_xcorr:
        _plot_xcorr(ax, data_sim_list, columns_sim_list, 'ground_reaction_forces', grf_names_sim[i], 1.0, data_exp['cycle'], values_exp, colors_sim)
    ax.set_xlim(0, 100)
    if i == 0:
      ax.set_ylabel(r'Force ($\% \text{BW}$)')

  spec = axes[2, 2].get_subplotspec()
  axes[2, 2].remove()
  sub_gs = spec.subgridspec(1, 2)
  ax_cycle_duration = fig.add_subplot(sub_gs[0])
  ax_stance_percentage = fig.add_subplot(sub_gs[1])

  bar_labels = list(labels_sim)
  bar_colors = list(colors_sim)
  cycle_durations_bar = [data_sim['cycle_duration'] for data_sim in data_sim_list]
  cycle_duration_errors = [np.nan] * len(data_sim_list)
  stance_percentages_bar = [data_sim['stance_percentage'] * 100 for data_sim in data_sim_list]
  stance_percentage_errors = [np.nan] * len(data_sim_list)
  if show_experimental:
    bar_labels.append(label_exp)
    bar_colors.append(to_rgba(color_exp, alpha=alpha_exp))
    cycle_durations_bar.append(np.nanmean(data_exp['cycle_duration']))
    cycle_duration_errors.append(np.nanstd(data_exp['cycle_duration']))
    stance_percentages_bar.append(np.nanmean(data_exp['stance_percentage']) * 100)
    stance_percentage_errors.append(np.nanstd(data_exp['stance_percentage']) * 100)

  x_positions = np.arange(len(bar_labels))

  ax_cycle_duration.bar(x_positions, cycle_durations_bar, color=bar_colors, yerr=cycle_duration_errors, capsize=capsize_exp)
  ax_cycle_duration.set_xticks(x_positions)
  ax_cycle_duration.set_xticklabels(bar_labels)
  ax_cycle_duration.tick_params(axis='x', rotation=90)
  ax_cycle_duration.set_ylabel(r'Cycle time ($\text{s}$)')

  ax_stance_percentage.bar(x_positions, stance_percentages_bar, color=bar_colors, yerr=stance_percentage_errors, capsize=capsize_exp)
  ax_stance_percentage.set_xticks(x_positions)
  ax_stance_percentage.set_xticklabels(bar_labels)
  ax_stance_percentage.tick_params(axis='x', rotation=90)
  ax_stance_percentage.set_ylabel(r'Stance pct. ($\%$)')

  muscle_titles = ['GMAX', 'HAMS', 'RF', 'VAS', 'GAS', 'TA']
  muscle_names_sim = ['glut_max_r', 'hamstrings_r', 'rect_fem_r', 'vasti_r', 'gastroc_r', 'tib_ant_r']
  muscle_names_exp = ['LGMAX', 'LBF', 'LRF', 'LVL', 'LMGAS', 'LTA']
  for i in range(6):
    ax = axes[3 + i // 3, i % 3]
    ax.set_title(muscle_titles[i])
    _plot_sim_curves(ax, data_sim_list, columns_sim_list, 'muscle_activations', muscle_names_sim[i], 1.0, colors_sim, labels_sim, linestyles_sim, linewidth_sim)
    if show_experimental:
      index_exp = columns_exp['muscle_activations'].index(muscle_names_exp[i])
      values_exp = data_exp['muscle_activations'][:, :, index_exp]
      _plot_exp_band(ax, data_exp['cycle'], values_exp, color_exp, alpha_exp, label_exp)
      if args.show_xcorr:
        _plot_xcorr(ax, data_sim_list, columns_sim_list, 'muscle_activations', muscle_names_sim[i], 1.0, data_exp['cycle'], values_exp, colors_sim)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    if i // 3 == 1:
      ax.set_xlabel(r'Cycle ($\%$)')
    if i % 3 == 0:
      ax.set_ylabel('Activation')

  handles_legend, labels_legend = axes[0, 0].get_legend_handles_labels()
  fig.legend(
    handles_legend, labels_legend,
    loc='outside lower center',
    ncols=len(labels_legend),
    frameon=False,
    borderaxespad=0.2,
    handlelength=1.5,
    columnspacing=1.2,
  )

  if args.show_xcorr and show_experimental:
    _add_headroom_for_xcorr(fig)

  fig.savefig(args.save_path)

  plt.show()

if __name__ == '__main__':
  plot_gait_data()
