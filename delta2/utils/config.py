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
import os
import json
import time
from shutil import copyfile, SameFileError
from pathlib import Path
import yaml
from absl import logging


def valid_config(config):
  ''' validation config '''
  del config
  return True


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
  valid_config(config)
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
