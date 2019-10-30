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
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta.serving.base_frozen_model import Evaluater 


@registers.serving.register
class EmoSpeechEvaluater(Evaluater):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str=None, mode=utils.INFER):
    self._config = config
    self._mode = mode
    model = config['serving']['model']
    super().__init__(model, gpu_str=gpu_str)
    input_name = config['serving']['inputs']
    output_name = config['serving']['outputs']

    self.audio_ph = self.graph.get_tensor_by_name(input_name)
    self.pred_valid = self.graph.get_tensor_by_name(output_name)

    self.inspect_ops()

  @property
  def config(self):
    ''' config '''
    return self._config

  #pylint: disable=too-many-locals
  def predict(self):
    ''' infer prediction results '''
    batch = 0
    #pylint: disable=invalid-name
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    solver_name = self.config['solver']['name']
    solver = registers.solver[solver_name](self.config)

    with self.graph.as_default():
      dataset = solver.input_fn(self._mode)()
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    try:
      while True:
        batch += 1
        logging.info("batch : {}".format(batch))

        features, labels = self.sess.run(next_element)
        inputs = features["inputs"]

        y_pred = self.sess.run(self.pred_valid, feed_dict={self.audio_ph: inputs})

        result = np.argmax(y_pred, axis=-1)
        for i, _ in enumerate(labels):
          #positive
          if labels[i] == 1:
            if labels[i] == result[i]:
              TP += 1
            else:
              FN += 1
          #Negative
          else:
            if labels[i] == result[i]:
              TN += 1
            else:
              FP += 1
    except tf.errors.OutOfRangeError:
      logging.info("Infer End")

    logging.info('TP {}'.format(TP))
    logging.info('TN {}'.format(TN))
    logging.info('FP {}'.format(FP))
    logging.info('FN {}'.format(FN))

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    logging.info('acc {}'.format(acc))
    logging.info('precision {}'.format(precision))
    logging.info('recall {}'.format(recall))
