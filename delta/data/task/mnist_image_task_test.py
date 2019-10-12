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
''' Test for mnist_image_task.py '''

import yaml
from absl import logging
import numpy as np
import tensorflow as tf

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register_v2

# pylint: disable=missing-docstring


class MnistImageTaskTest(tf.test.TestCase):
  ''' Unit test for AsrSeqTask. '''

  def setUp(self):
    ''' set up '''
    super().setUp()
    self.task_yaml = '''
    data:
      train:
        paths: null
        segments: null
      eval:
        paths: null
        segments: null
      infer:
        paths: null
        segments: null
      task:
        name: FashionMnistTask
        need_label: true
        dims: [28, 28, 1]
        batch_size: 512
    '''
    import_all_modules_for_register_v2()
    self.params = yaml.safe_load(self.task_yaml)
    logging.info(f'config: {self.params}')

  def test_dataset(self):
    task_conf = self.params['data']['task']
    task_name = task_conf['name']
    task_class = registers.task[task_name]
    task = task_class.params(task_conf).instantiate(utils.TRAIN)

    with self.cached_session(use_gpu=False) as sess:
      for images, labels in task.dataset(2):  # pylint: disable=bad-continuation
        self.assertDTypeEqual(images, np.float32)
        self.assertDTypeEqual(labels, np.int32)

        self.assertEqual(images.shape, (2, 28, 28, 1))
        self.assertEqual(labels.shape, (2,))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  tf.test.main()
