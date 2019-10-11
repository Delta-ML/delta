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
import sys
import tensorflow as tf
from absl import logging
from absl import app
import numpy as np

from delta import utils
from delta.utils import metrics as metrics_lib
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register
from delta.serving.base_frozen_model import FrozenModel


@registers.serving.register
class TextClsInfer(FrozenModel):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str=None, mode=utils.INFER):
    self._config = config
    self._mode = mode
    model = os.path.join(config['solver']['service']['model_path'],
                         config['solver']['service']['model_version'])
    super().__init__(model, gpu_str=gpu_str)

    # self.inspect_ops()

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

  def predict_one(self):
    feed_dict = {self.input_sentence: ["你好", "很开心"]}
    input_x = self.sess.run(self.input_x, feed_dict=feed_dict)
    logging.info(f"input_x: {input_x[0][:20]}, {np.shape(input_x)}")

    preds = self.sess.run(self.preds, feed_dict={self.input_x: input_x})
    logging.info(f"preds: {preds}")

    # preds = self.sess.run(self.preds, feed_dict=feed_dict)
    # logging.info(f"preds: {preds}")

    # while True:
    #   input_sent = input("Input sentence: ").strip()
    #   if input_sent == "q":
    #     sys.exit(0)
    #   feed_dict = {self.input_sentence: [input_sent]}
    #   # input_x = self.sess.run(self.input_x, feed_dict=feed_dict)
    #   # logging.info(f"input_x: {input_x[0][:20]}, {np.shape(input_x)}")
    #
    #   # preds = self.sess.run(self.preds, feed_dict={self.input_x: input_x})
    #   # logging.info(f"preds: {preds}")
    #
    #   preds = self.sess.run(self.preds, feed_dict=feed_dict)
    #   logging.info(f"preds: {preds}")


def main(_):
  ''' main func '''
  FLAGS = app.flags.FLAGS  #pylint: disable=invalid-name

  logging.info("config is {}".format(FLAGS.config))
  logging.info("mode is {}".format(FLAGS.mode))
  logging.info("gpu is {}".format(FLAGS.gpu))
  assert FLAGS.config, 'give a config.yaml'
  assert FLAGS.mode, 'give mode eval, infer or eval_and_infer'

  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu  #selects a specific device

  #create dataset
  if FLAGS.mode == 'infer':
    mode = utils.INFER
  else:
    mode = utils.EVAL

  # load config
  config = utils.load_config(FLAGS.config)

  # process config
  import_all_modules_for_register()
  solver_name = config['solver']['name']
  solver = registers.solver[solver_name](config)
  config = solver.config

  eval_obj = TextClsInfer(config, gpu_str=FLAGS.gpu, mode=mode)
  eval_obj.predict_one()


def define_flags():
  ''' define flags for evaluator'''
  app.flags.DEFINE_string('config', 'egs/mock_text_cls_data/text_cls/v1/config/han-cls.yml', help='config path')
  app.flags.DEFINE_string('mode', 'eval', 'eval, infer, eval_and_infer')
  # The GPU devices which are visible for current process
  app.flags.DEFINE_string('gpu', '', 'gpu number')


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_flags()
  app.run(main)
  logging.info("OK. Done!")
