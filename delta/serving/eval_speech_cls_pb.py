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
''' Frozen model Evaluater'''
import numpy as np
from absl import logging

import delta.compat as tf
from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta.serving.base_frozen_model import Evaluater


class ClsMetric:

  def __init__(self):
    self.TP = 0
    self.TN = 0
    self.FP = 0
    self.FN = 0

  def __call__(self, y_pred, y_true):
    for i, _ in enumerate(y_true):
      #positive
      if y_true[i] == 1:
        if y_true[i] == y_pred[i]:
          self.TP += 1
        else:
          self.FN += 1
      #Negative
      else:
        if y_true[i] == y_pred[i]:
          self.TN += 1
        else:
          self.FP += 1

  def result(self, log_verbosity=False):
    if log_verbosity:
      logging.info('TP {}'.format(self.TP))
      logging.info('TN {}'.format(self.TN))
      logging.info('FP {}'.format(self.FP))
      logging.info('FN {}'.format(self.FN))
    acc = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    precision = self.TP / (self.TP + self.FP)
    recall = self.TP / (self.TP + self.FN)
    return acc, precision, recall


class SpeechEvaluater(Evaluater):
  ''' base evaluater '''

  def __init__(self, config, gpu_str=None, mode=utils.INFER):
    self._config = config
    self._mode = mode
    model = config['serving']['model']
    super().__init__(model, gpu_str=gpu_str)
    input_name = config['serving']['inputs']
    output_name = config['serving']['outputs']

    self.audio_ph = self.graph.get_tensor_by_name(input_name)
    self.pred_valid = self.graph.get_tensor_by_name(output_name)

    self.metric = ClsMetric()

    self.inspect_ops()
    self.build()

  @property
  def config(self):
    ''' config '''
    return self._config

  def build(self):
    ''' build graph '''
    solver_name = self.config['solver']['name']
    solver = registers.solver[solver_name](self.config)

    with self.graph.as_default():
      dataset = solver.input_fn(self._mode)()
      iterator = dataset.make_one_shot_iterator()
      self.next_element = iterator.get_next()

  def postproc(self, pred, features=None):
    ''' prost processing '''
    result = np.argmax(pred, axis=-1)
    return result

  def run(self):
    ''' featch predictions '''
    features, y_true = self.sess.run(self.next_element)
    inputs = features["inputs"]
    pred = self.sess.run(self.pred_valid, feed_dict={self.audio_ph: inputs})
    y_pred = self.postproc(pred, features=features)
    return y_pred, y_true


@registers.serving.register
class EmoSpeechEvaluater(SpeechEvaluater):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str, mode):
    super().__init__(config, gpu_str, mode)

  def predict(self):
    ''' infer prediction results and/or log metrics '''
    batch = 0
    try:
      while True:
        batch += 1
        logging.info("process {} batches".format(batch))
        y_pred, y_true = self.run()
        if self._mode == utils.EVAL:
          self.metric(y_pred, y_true)
    except tf.errors.OutOfRangeError:
      logging.info("Process End")

    if self._mode == utils.EVAL:
      acc, precision, recall = self.metric.result()
      logging.info('acc {}'.format(acc))
      logging.info('precision {}'.format(precision))
      logging.info('recall {}'.format(recall))


@registers.serving.register
class SpkSpeechEvaluater(SpeechEvaluater):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str, mode):
    super().__init__(config, gpu_str, mode)

    postproc_name = self.config['solver']['postproc']['name']
    self.post_fn = registers.postprocess[postproc_name](self.config)

  def postproc(self, pred, features=None):
    ''' prost processing '''
    self.post_fn(pred)
    return

  def run(self):
    ''' featch predictions '''

    def gen():
      features, y_true = self.sess.run(self.next_element)
      inputs = features["inputs"]
      pred = self.sess.run(self.pred_valid, feed_dict={self.audio_ph: inputs})
      features.update({'embeddings': pred})
      return features

    class Iter:

      def __iter__(self):
        return self

      def __next__(self):
        return gen()

    self.postproc(Iter())
    return None, None

  def predict(self):
    ''' extract speaker embedding '''
    batch = 0
    try:
      while True:
        batch += 1
        logging.info("process {} batches".format(batch))
        self.run()
    except tf.errors.OutOfRangeError:
      logging.info("Process End")
