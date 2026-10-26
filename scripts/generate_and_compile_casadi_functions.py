import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

def _load_model(mjcf_model_path):
  model = pin.buildModelFromMJCF(mjcf_model_path)

  cmodel = cpin.Model(model)
  cdata = cmodel.createData()

  return cmodel, cdata

def _get_coordinate_index_map(mjcf_model_path, coordinate_names_path):
  tree = ET.parse(mjcf_model_path)
  root = tree.getroot()

  worldbody = root.find('worldbody')

  pin_coordinate_names = [joint.get('name') for joint in worldbody.iter('joint')]

  with open(coordinate_names_path, 'r') as f:
    orig_coordinate_names = json.load(f)

  orig_to_pin_indices = [orig_coordinate_names.index(name) for name in pin_coordinate_names]
  pin_to_orig_indices = [pin_coordinate_names.index(name) for name in orig_coordinate_names]

  return orig_to_pin_indices, pin_to_orig_indices

def _generate_and_compile_casadi_function(f, save_directory_path, save_base_name):
  jac_f = f.jacobian()
  jac_jac_f = jac_f.jacobian()
  jac_jac_jac_f = jac_jac_f.jacobian()

  jac_jac_jac_f_dummy = ca.Function(jac_jac_jac_f.name(), [ca.SX(*jac_jac_jac_f.size_in(i)) for i in range(jac_jac_jac_f.n_in())], [ca.SX(*jac_jac_jac_f.size_out(i)) for i in range(jac_jac_jac_f.n_out())])

  C = ca.CodeGenerator('tmp.c')
  C.add(f)
  C.add(jac_f)
  C.add(jac_jac_f)
  C.add(jac_jac_jac_f_dummy)
  C.generate()

  system_name = platform.system()

  if system_name == 'Windows':
    save_path = Path(save_directory_path) / f'{save_base_name}.dll'
    compile_command = ['gcc', '-shared', '-O2', 'tmp.c', '-o', save_path]

  else:
    save_path = Path(save_directory_path) / f'lib{save_base_name}.so'
    compile_command = ['gcc', '-shared', '-fPIC', '-O2', 'tmp.c', '-o', save_path]

  subprocess.run(compile_command)

  os.remove('tmp.c')

def _generate_and_compile_contact_kinematics_functions(mjcf_model_path, model_dir, save_dir):
  cmodel, cdata = _load_model(mjcf_model_path)

  orig_to_pin_indices, pin_to_orig_indices = _get_coordinate_index_map(mjcf_model_path, model_dir / 'coordinate_names.json')

  with open(model_dir / 'contact_names.json', 'r') as f:
    contact_names = json.load(f)

  with open(model_dir / 'contact_parameters.json', 'r') as f:
    contact_parameters = json.load(f)

  q_orig = ca.SX.sym('q', cmodel.nq)
  v_orig = ca.SX.sym('v', cmodel.nv)

  q_pin = q_orig[orig_to_pin_indices]
  v_pin = v_orig[orig_to_pin_indices]

  cpin.forwardKinematics(cmodel, cdata, q_pin, v_pin)
  cpin.updateFramePlacements(cmodel, cdata)

  for contact_name in contact_names:
    mobilized_body_name = contact_parameters[contact_name]['mobilized_body_name']
    mobilized_body_frame_id = cmodel.getFrameId(mobilized_body_name)

    T_mobilized_body = cdata.oMf[mobilized_body_frame_id]

    nu_mobilized_body_in_local = cpin.getFrameVelocity(cmodel, cdata, mobilized_body_frame_id, cpin.ReferenceFrame.LOCAL)

    J_mobilized_body_pin = cpin.computeFrameJacobian(cmodel, cdata, q_pin, mobilized_body_frame_id, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_mobilized_body_orig = J_mobilized_body_pin[:, pin_to_orig_indices]

    base_name = f'f_{contact_name}_kinematics'
    f = ca.Function(base_name, [q_orig, v_orig], [T_mobilized_body.np, nu_mobilized_body_in_local.np, J_mobilized_body_orig], ['q', 'v'], ['T_mobilized_body', 'nu_mobilized_body_in_local', 'J_mobilized_body'])

    _generate_and_compile_casadi_function(f, save_dir, base_name)

def _generate_and_compile_skeletal_dynamics_function(mjcf_model_path, model_dir, save_dir):
  cmodel, cdata = _load_model(mjcf_model_path)

  orig_to_pin_indices, pin_to_orig_indices = _get_coordinate_index_map(mjcf_model_path, model_dir / 'coordinate_names.json')

  q_osim = ca.SX.sym('q', cmodel.nq)
  v_osim = ca.SX.sym('v', cmodel.nv)
  tau_osim = ca.SX.sym('tau', cmodel.nv)

  q_pin = q_osim[orig_to_pin_indices]
  v_pin = v_osim[orig_to_pin_indices]
  tau_pin = tau_osim[orig_to_pin_indices]

  a_pin = cpin.aba(cmodel, cdata, q_pin, v_pin, tau_pin)

  a_osim = a_pin[pin_to_orig_indices]

  base_name = 'f_skeletal_dynamics'
  f = ca.Function(base_name, [q_osim, v_osim, tau_osim], [a_osim], ['q', 'v', 'tau'], ['a'])

  _generate_and_compile_casadi_function(f, save_dir, base_name)

def generate_and_compile_casadi_functions():
  parser = argparse.ArgumentParser()
  parser.add_argument('mjcf_model_path', type=Path)
  parser.add_argument('model_dir', type=Path)
  parser.add_argument('save_dir', type=Path)
  args = parser.parse_args()

  _generate_and_compile_contact_kinematics_functions(args.mjcf_model_path, args.model_dir, args.save_dir)
  _generate_and_compile_skeletal_dynamics_function(args.mjcf_model_path, args.model_dir, args.save_dir)

if __name__ == '__main__':
  generate_and_compile_casadi_functions()
