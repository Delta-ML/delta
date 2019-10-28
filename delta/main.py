#!/usr/bin/env python
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

"""Main entrance of the program."""

import random
import numpy as np
import delta.compat as tf
from absl import flags
from absl import app
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register


def define_flags():
  """Define flags for the program."""
  flags.DEFINE_string('config', '', 'config path')
  flags.DEFINE_string(
      'cmd', '',
      'train, eval, infer, train_and_eval, export_model, gen_feat, gen_cmvn')
  flags.DEFINE_bool('test', 'False', 'run all unit test')
  flags.DEFINE_bool('dry_run', 'False', 'dry run, to no save file')
  flags.DEFINE_bool('log_debug', 'False', 'logging debug switch')


FLAGS = flags.FLAGS


def set_seed(config):
  """Set the random seed."""
  random_seed = config['solver']['run_config']['tf_random_seed']
  random.seed(random_seed)
  np.random.seed(random_seed)
  tf.set_random_seed(random_seed)


def main(argv):
  """
    main function
  """
  # pylint: disable=unused-argument
  # load config
  config = utils.load_config(FLAGS.config)
  utils.set_logging(FLAGS.log_debug, config)

  utils.copy_config(FLAGS.config, config)
  set_seed(config)

  logging.info("Loading all modules ...")
  import_all_modules_for_register(config)

  solver_name = config['solver']['name']
  solver = registers.solver[solver_name](config)

  # config after process
  config = solver.config

  task_name = config['data']['task']['name']
  task_class = registers.task[task_name]

  logging.info("CMD: {}".format(FLAGS.cmd))
  if FLAGS.cmd == 'train':
    solver.train()
  elif FLAGS.cmd == 'train_and_eval':
    solver.train_and_eval()
  elif FLAGS.cmd == 'eval':
    solver.eval()
  elif FLAGS.cmd == 'infer':
    solver.infer(yield_single_examples=False)
  elif FLAGS.cmd == 'export_model':
    solver.export_model()
  elif FLAGS.cmd == 'gen_feat':
    assert config['data']['task'][
        'suffix'] == '.npy', 'wav does not need to extractor feature'
    paths = []
    for mode in [utils.TRAIN, utils.EVAL, utils.INFER]:
      paths += config['data'][mode]['paths']
    task = task_class(config, utils.INFER)
    task.generate_feat(paths, dry_run=FLAGS.dry_run)
  elif FLAGS.cmd == 'gen_cmvn':
    logging.info(
        '''using infer pipeline to compute cmvn of train_paths, and stride must be 1'''
    )
    paths = config['data'][utils.TRAIN]['paths']
    segments = config['data'][utils.TRAIN]['segments']
    config['data'][utils.INFER]['paths'] = paths
    config['data'][utils.INFER]['segments'] = segments
    task = task_class(config, utils.INFER)
    task.generate_cmvn(dry_run=FLAGS.dry_run)
  else:
    raise ValueError("Not support command: {}.".format(FLAGS.cmd))


def entry():
  define_flags()
  logging.info("Deep Language Technology Platform start...")
  app.run(main)
  logging.info("OK. Done!")


if __name__ == '__main__':
  entry()
