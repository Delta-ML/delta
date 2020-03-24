# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' configration utils'''
from typing import List, Union
import os
import json
import time
from shutil import copyfile, SameFileError
from pathlib import Path
import yaml
from absl import logging


def valid_config(config):
  ''' validation config'''
  del config
  return True


def config_join_project_path(project_dir: str, config: dict,
                             key_path: List[Union[str, int]]):
  """join project dir on a path"""
  d = config
  try:
    for k in key_path[:-1]:
      d = d[k]
    original_path = d[key_path[-1]]
  except KeyError as e:
    logging.warning(f"key_path: {key_path} not found!")
    raise KeyError(repr(e))
  if isinstance(original_path, list):
    d[key_path[-1]] = [os.path.join(project_dir, p) for p in original_path]
  elif isinstance(original_path, str):
    d[key_path[-1]] = os.path.join(project_dir, original_path)
  else:
    logging.warning(f"key_path: {key_path} error.")
    raise TypeError("path is not str or list!")

def config_join_project_dir(config):
  """operations after the config been loaded."""
  if 'data' not in config or "project_dir" not in config['data']:
    return
  project_dir = config['data']["project_dir"]
  file_key_paths = [['data', 'task', 'preparer', 'done_sign'],
                    ['data', 'task', 'text_vocab'],
                    ['data', 'task', 'label_vocab'],
                    ['solver', 'service', 'model_path'],
                    ['solver', 'saver', 'model_path']]

  for data_type in ['train', 'eval', 'infer']:
    if isinstance(config['data'][data_type]['paths'], dict):
      for k in config['data'][data_type]['paths']:
        file_key_paths.append(['data', data_type, 'paths', k])
    else:
      file_key_paths.append(['data', data_type, 'paths'])

  if isinstance(config['solver']['metrics'], dict):
    metric = config['solver']['metrics']
    if "target_file" in metric:
      file_key_paths.append(['solver', 'metrics', 'target_file'])
    if "text_vocab" in metric:
      file_key_paths.append(['solver', 'metrics', 'text_vocab'])
    if "res_file" in metric:
      file_key_paths.append(['solver', 'metrics', 'res_file'])
    for j, cal in enumerate(metric['cals']):
      if cal['arguments'] is not None and 'label_vocab_path' in cal['arguments']:
        file_key_paths.append(['solver', 'metrics', 'cals', j, 'arguments', 'label_vocab_path'])
  else:
    for i, metric in enumerate(config['solver']['metrics']):
      for j, cal in enumerate(metric['cals']):
        if cal['arguments'] is not None and 'label_vocab_path' in cal['arguments']:
          file_key_paths.append(['solver', 'metrics', i, 'cals', j, 'arguments', 'label_vocab_path'])

  if isinstance(config['solver']['postproc'], list):
    for i,postproc in enumerate(config['solver']['postproc']):
      file_key_paths.append(['solver', 'postproc', i, 'res_file'])
  else:
    file_key_paths.append(['solver', 'postproc', 'res_file'])

  for file_key_path in file_key_paths:
    config_join_project_path(project_dir, config, file_key_path)


def load_config(config_path):
  ''' load config from file '''
  if isinstance(config_path, Path):
    config_path = str(config_path)

  with open(config_path, 'r') as f:  #pylint: disable=invalid-name
    if config_path.endswith('yml') or config_path.endswith('yaml'):
      config = yaml.load(f, Loader=yaml.SafeLoader)
    elif config_path.endswith('json'):
      config = json.load(f)
  # check config
  # valid_config(config)
  config_join_project_dir(config)
  return config


def copy_config(config_path, config):
  ''' copy config file to ckpt dirctory '''
  if isinstance(config_path, Path):
    config_path = str(config_path)
  config_name = os.path.basename(config_path)
  save_config_path = os.path.join(config["solver"]["saver"]["model_path"],
                                  config_name)
  logging.info("Saving config file to {}".format(save_config_path))
  try:
    copyfile(config_path, save_config_path)
  except SameFileError:
    pass

  with open(config_path, 'r') as f:
    logging.info("Config:")
    logging.info(f.read())
  return config


def save_config(config, config_path):
  ''' save config to file '''
  if isinstance(config_path, Path):
    config_path = str(config_path)

  with open(config_path, 'w') as f:  #pylint: disable=invalid-name
    if config_path.endswith('yml') or config_path.endswith('yaml'):
      yaml.dump(config, f)
    elif config_path.endswith('json'):
      json.dump(config, f)


def setdefault_config(config):
  ''' set default config '''
  # This function only sets up those most commonly used parameters
  # which may not change frequently.

  config.setdefault('tfenv', dict())
  tfconf = config['tfenv']
  tfconf.setdefault('allow_soft_placement', True)
  tfconf.setdefault('log_device_placement', True)
  tfconf.setdefault('intra_op_parallelism_threads', 10)
  tfconf.setdefault('inter_op_parallelism_threads', 10)
  tfconf.setdefault('allow_growth', False)
  tfconf.setdefault('embedding_trainable', True)
  tfconf.setdefault('use_pretrained_embedding', False)

  model_conf = config['model']
  optimizer_conf = model_conf['optimizer']
  optimizer_conf.setdefault('dropout_keep_prob', 0.0)

  data_conf = config['data']
  saver_conf = data_conf['saver']
  saver_conf.setdefault('max_to_keep', 50)
  saver_conf.setdefault('checkpoint_every', 500)
  saver_conf.setdefault('evaluate_on_dev_every', 500)
  timestamp = str(int(time.time()))
  out_dir = os.path.abspath(os.path.join('/tmp', 'tf_runs', timestamp))
  saver_conf.setdefault('outdir', out_dir)
  saver_conf.setdefault('resume_model_path', out_dir)
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  logging.info("Create temp output directory: {}\n".format(out_dir))

  data_conf.setdefault('outdir', out_dir)
  data_conf.setdefault('vocab', out_dir + '/vocab')
  data_conf.setdefault('label', out_dir + '/label')

  return config
