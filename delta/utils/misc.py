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
''' utils for delta '''
import os
from absl import logging

import numpy as np
import delta.compat as tf
#pylint: disable=no-name-in-module,no-member
from tensorflow.python.client import device_lib
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.util import nest

from delta.utils.register import registers


#pylint: disable=invalid-name
def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, _ in enumerate(static):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def len_to_mask(length, maxlen=None, dtype=tf.bool):
  ''' convert length to masking flag '''
  # (B) -> (B, T)
  mask = tf.sequence_mask(length, maxlen=maxlen, dtype=dtype)
  return mask


def len_to_padding(length, maxlen=None, dtype=tf.bool):
  ''' convert length to padding flag '''
  # (B) -> (B, T)
  return tf.cast(1 - len_to_mask(length, maxlen=maxlen, dtype=dtype), dtype)


def log_vars(prefix, variables):
  ''' logging TF varables metadata '''
  for var in variables:
    logging.info("{}: name: {} shape: {} device: {}".format(
        prefix, var.name, var.shape, var.device))


#pylint: disable=bad-continuation
def losses(config):
  ''' get loss object from register '''
  if 'distilling' in config['solver'] and config['solver']['distilling'][
      'enable']:
    loss_name = config['solver']['distilling']['loss']
  else:
    loss_name = config['solver']['optimizer']['loss']
  if isinstance(loss_name, list):
    _loss_fn = []
    for one_loss_name in loss_name:
      logging.info('loss == {}'.format(one_loss_name))
      _loss_fn.append(registers.loss[one_loss_name](config))
  else:
    logging.info('loss == {}'.format(loss_name))
    _loss_fn = registers.loss[loss_name](config)
  return _loss_fn


def task(config, mode):
  ''' get task object from register '''
  task_name = config['data']['task']["name"]
  logging.info("task == {}, mode == {}".format(task_name, mode))
  _task = registers.task[task_name](config, mode)
  return _task


def model(config):
  ''' get model object from register '''
  classname = config['model']['name']
  logging.info("model == {}".format(classname))
  # Model initialization
  _model = registers.model[classname](config)
  return _model


def gpu_device_names():
  '''
  :returns, list of gpu device name, num of gpus
  '''
  devices = []
  for x in device_lib.list_local_devices():  #pylint: disable=invalid-name
    if x.device_type == 'GPU':
      devices.append(tf.compat.as_text(x.name))
  return devices, len(devices)


def tf_version_satisfy(target_version_str):
  '''
  A convenient function to check TF version.

  Args:
    target_version_str: a string, e.g. '1.14', '1.12.0'

  Returns:
    True if TF version is greater or equal than target version.
  '''
  current_version_str = tf.__version__
  current_version = [int(num) for num in current_version_str.split('.')]
  target_version = [int(num) for num in target_version_str.split('.')]
  satisfied = current_version >= target_version
  return satisfied


def get_distribution_strategy(num_gpus, all_reduce_alg='nccl'):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.distribute.NcclAllReduce for available algorithms.
      If None, Strategy will using nccl as default.

  Returns:
    tf.distribute.Strategy object.
  """
  if num_gpus == 0:  #pylint: disable=no-else-return
    return tf.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    return tf.distribute.MirroredStrategy(devices=None, cross_device_ops=None)


def per_device_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.

  Returns:
    Batch size per device.

  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ("When running with multiple GPUs, batch size "
           "must be a multiple of the number of available GPUs. Found {} "
           "GPUs with a batch size of {}; try --batch_size={} instead.").format(
               num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)


#pylint: disable=too-many-arguments
def generate_synthetic_data(input_shape,
                            input_value=0,
                            input_dtype=None,
                            label_shape=None,
                            label_value=0,
                            label_dtype=None,
                            nepoch=None):
  """Create a repeating dataset with constant values.

  Args:
    input_shape: a tf.TensorShape object or nested tf.TensorShapes. The shape of
      the input data.
    input_value: Value of each input element.
    input_dtype: Input dtype. If None, will be inferred by the input value.
    label_shape: a tf.TensorShape object or nested tf.TensorShapes. The shape of
      the label data.
    label_value: Value of each input element.
    label_dtype: Input dtype. If None, will be inferred by the target value.
    nepoch: num of epochs. If None, will repeat forever.

  Returns:
    Dataset of tensors or tuples of tensors (if label_shape is set).
  """
  # TODO(kathywu): Replace with SyntheticDataset once it is in contrib.
  element = input_element = nest.map_structure(
      lambda s: tf.constant(input_value, input_dtype, s), input_shape)

  if label_shape:
    label_element = nest.map_structure(
        lambda s: tf.constant(label_value, label_dtype, s), label_shape)
    element = (input_element, label_element)

  return tf.data.Dataset.from_tensors(element).repeat(nepoch)


def metric_smaller(best_eval_result,
                   current_eval_result,
                   default_key=metric_keys.MetricKeys.AUC):
  """Compares two evaluation results and returns true if the 2nd one is smaller.
              Both evaluation results should have the values for MetricKeys.LOSS, which are
              used for comparison.
              Args:
                best_eval_result: best eval metrics.
                current_eval_result: current eval metrics.
                default_key: metric_keys.MericKeys
              Returns:
                True if the loss of current_eval_result is smaller; otherwise, False.
              Raises:
               ValueError: If input eval result is None or no loss is available.
        """
  if not best_eval_result or default_key not in best_eval_result:
    raise ValueError(
        'best_eval_result cannot be empty or no loss is found in it.')

  if not current_eval_result or default_key not in current_eval_result:
    raise ValueError(
        'current_eval_result cannot be empty or no loss is found in it.')
  return best_eval_result[default_key] > current_eval_result[default_key]


def listdir(path):
  ''' generate files of path '''
  for filename in os.listdir(path):
    yield os.path.join(path, filename)


def walk(path='.', depth=None):
  """
    recursively walk directory to specified depth
    :param path: (str) the base path to start walking from
    :param depth: (None or int) max. recursive depth, None = no limit
    :yields: (str) filename, including path
    """
  if depth and depth == 1:
    for filename in listdir(path):
      yield filename
  else:
    top_pathlen = len(path) + len(os.path.sep)
    for dirpath, dirnames, filenames in os.walk(path):
      dirlevel = dirpath[top_pathlen:].count(os.path.sep)
      if depth and dirlevel >= depth:
        dirnames[:] = []
      else:
        for filename in filenames:
          yield os.path.join(dirpath, filename)


# 'train', 'eval', 'infer'
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
INFER = tf.estimator.ModeKeys.PREDICT
PAD_IDX = 0
