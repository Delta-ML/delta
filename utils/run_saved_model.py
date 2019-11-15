#!/usr/bin/env python3
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
''' Saved & Frozen & Checkpoint model Evaluater'''
import os
from absl import logging
from absl import flags
from absl import app

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register


def main(_):
  ''' main func '''
  FLAGS = app.flags.FLAGS  #pylint: disable=invalid-name
  logging.info("config: {}".format(FLAGS.config))
  logging.info("mode: {}".format(FLAGS.mode))
  logging.info("gpu_visible: {}".format(FLAGS.gpu))
  assert FLAGS.config, 'pls give a config.yaml'
  assert FLAGS.mode, 'pls give mode [eval|infer|eval_and_infer]'
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu  #selects a specific device

  #create dataset
  mode = utils.INFER if FLAGS.mode == 'infer' else utils.EVAL

  # load config
  config = utils.load_config(FLAGS.config)

  # process config
  import_all_modules_for_register()
  solver_name = config['solver']['name']
  logging.info(f"sovler: {solver_name}")
  solver = registers.solver[solver_name](config)
  config = solver.config

  # Evaluate
  evaluate_name = config['serving']['name']
  logging.info(f"evaluate: {evaluate_name}")
  evaluate = registers.serving[evaluate_name](
      config, gpu_str=FLAGS.gpu, mode=mode)

  if FLAGS.debug:
    evaluate.debug()
  evaluate.predict()


def define_flags():
  ''' define flags for evaluator'''
  # The GPU devices which are visible for current process
  flags.DEFINE_string('gpu', '', 'same to CUDA_VISIBLE_DEVICES')
  flags.DEFINE_string('config', None, help='path to yaml config file')
  flags.DEFINE_enum('mode', 'eval', ['eval', 'infer', 'eval_and_infer'],
                    'eval or infer')
  flags.DEFINE_bool('debug', False, 'debug mode')
  # https://github.com/abseil/abseil-py/blob/master/absl/flags/_validators.py#L330
  flags.mark_flags_as_required(['config', 'mode'])


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_flags()
  app.run(main)
  logging.info("OK. Done!")
