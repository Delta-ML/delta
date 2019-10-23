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
''' kws task unittest'''
import os
from pathlib import Path
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register


class KwsClsTaskTest(tf.test.TestCase):
  ''' kws task test'''

  def setUp(self):
    super().setUp()
    import_all_modules_for_register()
    '''
    package_root = Path(PACKAGE_ROOT_DIR)
    config_file = main_root.joinpath('delta/config/kws-cls/kws_speech_cls.yml')
    config = utils.load_config(config_file)

    solver_name = config['solver']['name']
    self.solver = registers.solver[solver_name](config)

    # config after process
    self.config = self.solver.config

    self.mode = utils.EVAL

    task_name = self.config['data']['task']['name']
    self.task = registers.task[task_name](self.config, self.mode)

    self.dataset = self.task.dataset(self.mode, 25, 0)
    self.iterator = self.dataset.make_one_shot_iterator()
    self.one_element = self.iterator.get_next()
    '''

  def tearDown(self):
    ''' tear down '''

  def test_dataset(self):
    ''' dataset unittest'''
    pass
    '''
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      for _ in range(2):
        output = sess.run(self.one_element)
        logging.info(output)
        logging.info("output: {} {}".format(output.shape, output.dtype))
    '''


if __name__ == '__main__':
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
