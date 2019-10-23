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
"""Model related utilities."""

import os
import numpy as np
import hurry.filesize as hfsize
from absl import logging
import delta.compat as tf
# pylint: disable=no-name-in-module
from tensorflow.python.framework import graph_util


def print_ops(graph, prefix=''):
  """Print tensorflow operations in a graph."""
  for operator in graph.get_operations():
    logging.info('{} : op name: {}'.format(prefix, operator.name))


def log_vars(prefix, variables):
  """Print tensorflow variables."""
  for var in variables:
    logging.info("{}: name: {} shape: {} device: {}".format(
        prefix, var.name, var.shape, var.device))


def model_size(variables):
  """Get model size."""
  total_params = sum(
      [np.prod(var.shape.as_list()) * var.dtype.size for var in variables])
  return hfsize.size(total_params, system=hfsize.alternative)


def save(saver, session, ckpt_dir, ckpt_name="best"):
  """Save model checkpoint."""
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    logging.info("make dir: {}".format(ckpt_dir))
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  logging.info('[*] saving checkpoints to {}...'.format(ckpt_path))
  saver.save(session, ckpt_path)


def load(saver, session, ckpt_dir, ckpt_name="best"):
  """Load model from a checkpoint."""
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  logging.info('[*] Loading checkpoints from {}...'.format(ckpt_path))
  try:
    saver.restore(session, ckpt_path)
  except Exception as exception:
    logging.info(exception)
    logging.info("check ckpt file path !!!")
    raise exception


def get_sess_config(gpu_str=None):
  """generate a session config proto"""
  config = tf.ConfigProto()

  # pylint: disable=no-member
  if gpu_str is None:
    config.gpu_options.visible_device_list = ''
  else:
    config.gpu_options.visible_device_list = gpu_str
    config.gpu_options.allow_growth = True
  return config


def get_session(sess_config):
  """load a new session"""
  return tf.Session(config=sess_config)


def load_frozen_graph(frozen_graph_filename, print_op=False):
  """load a graph from protocol buffer file"""
  # We load the protobuf file from the disk and parse it to retrieve the
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as in_f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(in_f.read())

  # Then, we import the graph_def into a new Graph and returns it
  with tf.Graph().as_default() as graph:  #pylint: disable=not-context-manager
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name="",
        op_dict=None,
        producer_op_list=None)
    if print_op:
      print_ops(graph, prefix='load_frozen_graph')
  return graph


def load_graph_session_from_ckpt(ckpt_path, sess_config, print_op=False):
  """load graph and session from checkpoint file"""
  graph = tf.Graph()
  with graph.as_default():  #pylint: disable=not-context-manager
    sess = get_session(sess_config)
    with sess.as_default():  #pylint: disable=not-context-manager
      # Load the saved meta graph and restore variables
      saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path))
      saver.restore(sess, ckpt_path)
    if print_op:
      print_ops(graph, prefix='load_graph_session_from_ckpt')
  return graph, sess


def load_graph_session_from_pb(pb_file, sess_config, print_op=False):
  """load graph and session from protocol buffer file"""
  graph = load_frozen_graph(pb_file, print_op)
  with graph.as_default():
    sess = get_session(sess_config)
  return graph, sess


def load_graph_session_from_saved_model(saved_model_dir,
                                        sess_config,
                                        print_op=False):
  """Load graph session from SavedModel"""
  if not tf.saved_model.maybe_saved_model_directory(saved_model_dir):
    raise ValueError("Not a saved model dir: {}".format(saved_model_dir))

  logging.info('saved model dir : {}'.format(saved_model_dir))
  graph = tf.Graph()
  with graph.as_default():  #pylint: disable=not-context-manager
    sess = get_session(sess_config)
    with sess.as_default():  #pylint: disable=not-context-manager
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 saved_model_dir)
      if print_op:
        print_ops(graph, prefix='load_graph_session_from_saved_model')
  return graph, sess


def frozen_graph_to_pb(outputs, frozen_graph_pb_path, sess, graph=None):
  """Freeze graph to a pb file."""
  if not isinstance(outputs, (list)):
    raise ValueError("Frozen graph: outputs must be list of output node name")

  if graph is None:
    graph = tf.get_default_graph()

  input_graph_def = graph.as_graph_def()
  logging.info("Frozen graph: len of input graph nodes: {}".format(
      len(input_graph_def.node)))

  # We use a built-in TF helper to export variables to constant
  output_graph_def = graph_util.convert_variables_to_constants(
      sess,
      input_graph_def,
      outputs,
  )

  logging.info("Frozen graph: len of output graph nodes: {}".format(
      len(output_graph_def.node)))  # pylint: disable=no-member

  with tf.gfile.GFile(frozen_graph_pb_path, "wb") as in_f:
    in_f.write(output_graph_def.SerializeToString())
