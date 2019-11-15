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
''' Frozen ASR model Evaluater'''
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils import metrics as metrics_lib
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta.serving.base_frozen_model import Evaluater


@registers.serving.register
class ASREvaluater(Evaluater):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str=None, mode=utils.INFER):
    self._config = config
    self._mode = mode
    model = config['serving']['model']
    super().__init__(model, gpu_str='0')

    self.inputs = self.graph.get_tensor_by_name(config['serving']['inputs'])
    self.input_length = self.graph.get_tensor_by_name(
        config['serving']['input_length'])
    self.pred_valid = self.graph.get_tensor_by_name(
        config['serving']['outputs'])

  @property
  def config(self):
    ''' config '''
    return self._config

  #pylint: disable=too-many-locals
  def predict(self):
    ''' infer prediction results '''
    batch = 0

    solver_name = self.config['solver']['name']
    solver = registers.solver[solver_name](self.config)

    with self.graph.as_default():
      dataset, _ = solver.input_data(self._mode)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

    target_seq_list, predict_seq_list = [], []
    num_input_samples, num_processed_samples = 0, 0
    try:
      while True:
        batch += 1
        logging.info("batch : {}".format(batch))

        features, _ = self.sess.run(next_element)
        inputs = features["inputs"]
        input_length = features["input_length"]
        y_true_valid = features["targets"]
        num_input_samples += input_length.shape[0]
        logging.info('The size of the INFER Set increased to {}'.format(
            num_input_samples))

        validate_feed = {self.inputs: inputs, self.input_length: input_length}
        y_pred_valid = self.sess.run(self.pred_valid, feed_dict=validate_feed)
        num_processed_samples += y_pred_valid.shape[0]
        logging.info(
            'A total of {} samples has been successfully processed'.format(
                num_processed_samples))

        target_seq_list.extend(y_true_valid.tolist())
        predict_seq_list.extend(y_pred_valid.tolist())

    except tf.errors.OutOfRangeError:
      logging.info("Infer End")

    token_errors = metrics_lib.token_error(
        predict_seq_list=predict_seq_list,
        target_seq_list=target_seq_list,
        eos_id=0)
    logging.info('Token ERR: {}'.format(token_errors))
