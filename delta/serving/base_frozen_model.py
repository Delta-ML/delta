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
''' base frozen model '''
import os
import sys
import abc
from absl import logging
import delta.compat as tf

from delta import utils


class ABCFrozenModel(metaclass=abc.ABCMeta):
  '''Abstract class of FrozenModel'''

  @abc.abstractmethod
  def init_session(self, model, gpu_str):
    ''' init_session '''
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def graph(self):
    ''' graph '''
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def sess(self):
    ''' sess '''
    raise NotImplementedError()


class FrozenModel(ABCFrozenModel):
  '''FrozenModel'''

  def __init__(self, model, gpu_str=None):
    '''
     model: saved model dir, ckpt dir or frozen_graph_pb path
     gpu_str: list of gpu devices. e.g. '' for cpu, '0,1' for gpu 0,1
    '''
    self.init_session(model, gpu_str)

  def init_session(self, model, gpu_str):
    # The config for CPU usage
    config = tf.ConfigProto()
    if not gpu_str:
      config.gpu_options.visible_device_list = ''  # pylint: disable=no-member
    else:
      config.gpu_options.visible_device_list = gpu_str  # pylint: disable=no-member
      config.gpu_options.allow_growth = True  # pylint: disable=no-member

    #check model dir
    if os.path.isdir(model):
      self._graph = tf.Graph()

      if tf.saved_model.maybe_saved_model_directory(model):
        #saved model
        logging.info('saved model dir : {}'.format(model))
        self._sess = tf.Session(graph=self._graph, config=config)
        tf.saved_model.loader.load(self._sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   model)
      else:
        #checkpoint
        self._sess = tf.Session(
            graph=self._graph,
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
        ckpt_path = tf.train.latest_checkpoint(model)
        # self._graph, self._sess = utils.load_graph_session_from_ckpt(ckpt_path)
        model = ckpt_path + '.meta'
        logging.info("meta : {}".format(model))
        saver = tf.train.import_meta_graph(model)
        saver.restore(self._sess, ckpt_path)

    else:
      if not os.path.exists(model):
        logging.info('{}, is not exist'.format(model))
        logging.info("frozen_graph : {} not exist".format(model))
        sys.exit(0)

      #frozen graph pb
      frozen_graph = model
      logging.info('frozen graph pb : {}'.format(frozen_graph))
      self._graph = utils.load_frozen_graph(frozen_graph)
      self._sess = tf.Session(graph=self._graph, config=config)

  def inspect_ops(self):
    for op in self._graph.get_operations():
      logging.info(op.name)

  def debug(self):
    feed_dict = self.get_test_feed_dict()
    while True:
      tensor_name = input("Input debug tensor name: ").strip()
      if tensor_name == "q":
        sys.exit(0)
      try:
        debug_tensor = self.graph.get_tensor_by_name(tensor_name)
      except Exception as e:
        logging.error(e)
        continue
      res = self.sess.run(debug_tensor, feed_dict=feed_dict)
      logging.info(f"Result for tensor {tensor_name} is: {res}")

  @property
  def graph(self):
    return self._graph

  @property
  def sess(self):
    return self._sess
