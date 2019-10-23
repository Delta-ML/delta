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
"""Solver utilities."""

import os
import re
from pathlib import Path
import numpy as np
import delta.compat as tf
from absl import logging
import shutil

from delta import utils
from delta.utils import metrics


def get_checkpoint_dir(config):
  """Get the directory of the checkpoint."""
  model_path = config['solver']['saver']['model_path']
  checkpoint_dir = os.path.join(model_path, "model")
  return checkpoint_dir


def get_ckpt_state(config):
  """Get the checkpoint state."""
  checkpoint_dir = get_checkpoint_dir(config)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  return ckpt


def get_session_conf(config):
  """Get the config for the tensorflow session."""
  tfconf = config['solver']['run_config']
  session_conf = tf.ConfigProto(
      allow_soft_placement=tfconf['allow_soft_placement'],
      log_device_placement=tfconf['log_device_placement'],
      intra_op_parallelism_threads=tfconf['intra_op_parallelism_threads'],
      inter_op_parallelism_threads=tfconf['inter_op_parallelism_threads'],
      gpu_options=tf.GPUOptions(allow_growth=tfconf['allow_growth']))
  return session_conf


def to_saved_model(config, sess, inputs: dict, outputs: dict):
  """Save model to tensorflow SavedModel."""
  export_path_base = config["solver"]["service"]["model_path"]
  model_version = config["solver"]["service"]["model_version"]
  export_path = os.path.join(
      tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(model_version))
  export_path = os.path.abspath(export_path)
  logging.info('Exporting model to: {}'.format(export_path))
  if os.path.exists(export_path):
    files = [one.decode() for one in os.listdir(export_path) if isinstance(one, bytes)]
    if "variables" in files:
      cmd = input(f"Export directory already exists, and isn't empty. Overwrite? [y/n]").strip().lower()
      if cmd == "" or cmd == "y":
        shutil.rmtree(export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  # Build the signature_def_map.
  signature_def = tf.saved_model.predict_signature_def(inputs, outputs)
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={'infer': signature_def},
      strip_default_attrs=True)
  builder.save(as_text=True)
  logging.info('Done exporting!')


def save_infer_res(config, logits, preds):
  """Save the result of inference."""
  res_file = config["data"]["infer"]["res"]
  res_dir = os.path.dirname(res_file)
  if not os.path.exists(res_dir):
    os.makedirs(res_dir)
  logging.info("Save inference result to: {}".format(res_file))
  with open(res_file, "w") as in_f:
    for logit, pred in zip(logits, preds):
      in_f.write(" ".join(["{:.3f}".format(num) for num in logit]) +
                 "\t{}\n".format(pred))


def get_model_file(dir_name, file_name_pattern, mode, model_load_type,
                   specified_model_file_name):
  """
    Return model file according the specified model_load_type
    :param dir_name: the folder path where a file search will start
    :param file_name_pattern: the filename pattern that will be matched,
                              when searching model file with model_load_type=latest
    :param mode: which kind of command is performing [train, eval, infer]
    :param model_load_type: restore which kind of model [best, lastest, scratch, specific]
    :param specified_model_file_name: the model file which will be restored 
                                      with model_load_type=specific
    """
  assert model_load_type in (None, "best", "latest", "scratch", "specific")

  if model_load_type is None:
    logging.warning("The values of model_load_type is not specified.")
    model_load_type = "latest" if mode == utils.TRAIN else "best"
    logging.warning("For the {} command, model_load_type:{} is adopted.".format(
        mode, model_load_type))

  #model_load_type can not be 'scratch' when performing EVAL or INFER command
  if model_load_type == 'scratch' and mode != utils.TRAIN:
    model_load_type = "best"
    logging.warning(
        "The model_load_type cannot be scratch when performing {} command, and is changed to {}"
        .format(mode, model_load_type))

  #get the path of model file according the specificed model_load_type
  model_file_name = None
  if model_load_type == "specific":
    model_file_name = Path(dir_name).joinpath(specified_model_file_name)
    #the value of model_load_type will be changed to latest when specified_model_file_name is None
    if not model_file_name.exists():
      model_load_type = "latest"
      logging.warning(
          "The specified model file {} is not exist, model_load_type:{} is adopted"
          .format(model_file_name, model_load_type))

  if model_load_type == "latest":
    model_file_name = get_most_recently_modified_file_matching_pattern(
        dir_name, file_name_pattern)
  elif model_load_type == "best":
    model_file_name = Path(dir_name).joinpath('best_model.ckpt')

  #verify the existence of the file
  #model_file_name will be None when
  #     1.model_load_type=scratch
  #     2.no model_file is found with model_load_type=latest
  if model_file_name is None:
    logging.warning(
        'No model file is found in {} with model_load_type={}'.format(
            dir_name, model_load_type))
    if mode == utils.TRAIN:
      model_load_type = 'scratch'
      model_file_name = None
      logging.warning('The model will be trained with model_load_type:scratch')
    else:
      assert False, '{} END, since no model file is found'.format(mode)

  return model_load_type, model_file_name


def get_most_recently_modified_file_matching_pattern(dir_name,
                                                     file_name_pattern):
  """Return the most recently checkpoint file matching file_name_pattern"""
  file_name_regex = '^' + re.sub(r'{.*}', r'.*', file_name_pattern) + '$'

  tf_checkpoint_file = tf.train.latest_checkpoint(dir_name)
  if tf_checkpoint_file is not None and re.match(file_name_regex,
                                                 tf_checkpoint_file.name):
    return tf_checkpoint_file

  file_list = [
      file_name for file_name in Path(dir_name).iterdir()
      if re.match(file_name_regex, file_name.name)
  ]
  file_time_list = [single_file.stat().st_mtime for single_file in file_list]
  file_sort_by_time = np.argsort(file_time_list)
  latest_file = file_list[
      file_sort_by_time[-1]] if file_sort_by_time.shape[0] > 0 else None
  return latest_file


def run_metrics(config, y_preds, y_ground_truth, mode):
  """Run metrics for one output"""
  metcs = metrics.get_metrics(
      config=config, y_pred=y_preds, y_true=y_ground_truth)
  logging.info("Evaluation on %s:" % mode)
  if isinstance(metcs, list):
    for one_metcs in metcs:
      for key in sorted(one_metcs.keys()):
        logging.info(key + ":" + str(one_metcs[key]))
  else:
    for key in sorted(metcs.keys()):
      logging.info(key + ":" + str(metcs[key]))
