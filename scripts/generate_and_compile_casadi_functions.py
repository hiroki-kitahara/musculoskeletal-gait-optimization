import json
import os
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

MJCF_MODEL_PATH = 'models/2d_human_musculoskeletal_model/mjcf_model/2d_human_musculoskeletal_model.xml'

COORDINATE_NAMES_PATH = 'models/2d_human_musculoskeletal_model/coordinate_names.json'

CONTACT_NAMES_PATH = 'models/2d_human_musculoskeletal_model/contact_names.json'
CONTACT_PARAMETERS_PATH = 'models/2d_human_musculoskeletal_model/contact_parameters.json'

CONTACT_KINEMATICS_FUNCTIONS_SAVE_DIRECTORY_PATH = 'models/2d_human_musculoskeletal_model'
SKELETAL_DYNAMICS_FUNCTION_SAVE_DIRECTORY_PATH = 'models/2d_human_musculoskeletal_model'

def load_model():
  model = pin.buildModelFromMJCF(MJCF_MODEL_PATH)

  cmodel = cpin.Model(model)
  cdata = cmodel.createData()

  return cmodel, cdata

def get_coordinate_index_map():
  tree = ET.parse(MJCF_MODEL_PATH)
  root = tree.getroot()

  worldbody = root.find('worldbody')

  pin_coordinate_names = [joint.get('name') for joint in worldbody.iter('joint')]

  with open(COORDINATE_NAMES_PATH, 'r') as f:
    orig_coordinate_names = json.load(f)

  orig_to_pin_indices = [orig_coordinate_names.index(name) for name in pin_coordinate_names]
  pin_to_orig_indices = [pin_coordinate_names.index(name) for name in orig_coordinate_names]

  return orig_to_pin_indices, pin_to_orig_indices

def generate_and_compile_casadi_function(f, save_directory_path, save_base_name):
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

def generate_and_compile_contact_kinematics_functions():
  cmodel, cdata = load_model()

  orig_to_pin_indices, pin_to_orig_indices = get_coordinate_index_map()

  with open(CONTACT_NAMES_PATH, 'r') as f:
    contact_names = json.load(f)

  with open(CONTACT_PARAMETERS_PATH, 'r') as f:
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

    generate_and_compile_casadi_function(f, CONTACT_KINEMATICS_FUNCTIONS_SAVE_DIRECTORY_PATH, base_name)

def generate_and_compile_skeletal_dynamics_function():
  cmodel, cdata = load_model()

  orig_to_pin_indices, pin_to_orig_indices = get_coordinate_index_map()

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

  generate_and_compile_casadi_function(f, SKELETAL_DYNAMICS_FUNCTION_SAVE_DIRECTORY_PATH, base_name)

if __name__ == '__main__':
  generate_and_compile_contact_kinematics_functions()
  generate_and_compile_skeletal_dynamics_function()
