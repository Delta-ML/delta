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
import tensorflow as tf
from absl import logging

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


class DatasetInitializerHook(tf.train.SessionRunHook):
  def __init__(self, iterator, init_feed_dict):
    self._iterator = iterator
    self._init_feed_dict = init_feed_dict

  def begin(self):
    self._initializer = self._iterator.initializer

  def after_create_session(self, session, coord):
    del coord
    session.run(self._initializer, self._init_feed_dict)
