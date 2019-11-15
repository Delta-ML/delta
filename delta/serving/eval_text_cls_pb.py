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
''' Frozen text classification model Evaluater'''
import os
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta.serving.base_frozen_model import Evaluater


@registers.serving.register
class TextClsEvaluater(Evaluater):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str=None, mode=utils.INFER):
    self._config = config
    self._mode = mode
    model = os.path.join(config['solver']['service']['model_path'],
                         config['solver']['service']['model_version'])
    super().__init__(model, gpu_str=gpu_str)

    self.inspect_ops()

    self.input_sentence = self.graph.get_tensor_by_name(
        config['solver']['service']['input_sentence'])
    self.input_x = self.graph.get_tensor_by_name(
        config['solver']['service']['input_x'])
    self.score = self.graph.get_tensor_by_name(
        config['solver']['service']['score'])
    self.preds = self.graph.get_tensor_by_name(
        config['solver']['service']['preds'])

  @property
  def config(self):
    ''' config '''
    return self._config

  def get_test_feed_dict(self):
    return {self.input_sentence: ["你好", "很开心"]}

  def predict(self):
    feed_dict = self.get_test_feed_dict()

    input_x, score, preds = self.sess.run(
        [self.input_x, self.score, self.preds], feed_dict=feed_dict)
    logging.info(f"input_x: {input_x}")
    logging.info(f"preds: {preds}")
    logging.info(f"score: {score}")
