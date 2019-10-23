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
import os
import numpy as np
import delta.compat as tf
from absl import logging
from absl import app

from delta import utils
from delta.utils.register import registers
from delta.serving.base_frozen_model import FrozenModel


@registers.serving.register
class Evaluate(FrozenModel):
  ''' infer from forzen model '''

  def __init__(self, config, gpu_str=None, mode=utils.INFER):
    self._config = config
    self._mode = mode
    model = config['serving']['model']
    super().__init__(model, gpu_str='0')

    self.audio_ph = self.graph.get_tensor_by_name('inputs:0')
    self.pred_valid = self.graph.get_tensor_by_name('softmax_output:0')

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

        validate_feed = {
            self.audio_ph: inputs,
        }
        y_pred_valid = self.sess.run(self.pred_valid, feed_dict=validate_feed)
        result = np.argmax(y_pred_valid, axis=-1)

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
  solver_name = config['solver']['name']
  solver = registers.solver[solver_name](config)
  config = solver.config

  eval_obj = Evaluate(config, gpu_str=FLAGS.gpu, mode=mode)
  eval_obj.predict()


def define_flags():
  ''' define flags for evaluator'''
  app.flags.DEFINE_string('config', '', help='config path')
  app.flags.DEFINE_string('mode', '', 'eval, infer, eval_and_infer')
  app.flags.DEFINE_string('gpu', '0', 'gpu number')


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_flags()
  app.run(main)
  logging.info("OK. Done!")
