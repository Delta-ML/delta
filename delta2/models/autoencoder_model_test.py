# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for autoencoder model.py"""

import yaml
import numpy as np
import tensorflow as tf
from absl import logging

from delta import utils
from delta.utils.register import registers
from delta.utils.register import import_all_modules_for_register_v2

# pylint: disable=missing-docstring

class AEModelTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.task_yaml = '''
    model:
      name: AEModel 
      dims: [28, 28, 1]
      z_dim: 200
    '''
    import_all_modules_for_register_v2()

    self.config = yaml.safe_load(self.task_yaml)
    logging.info(f'config: {self.config}')

  def test_build_model(self):
    with self.cached_session(use_gpu=False) as sess:
      self.assertTrue(tf.executing_eagerly())

      model_conf = self.config['model']
      model_name = model_conf['name']
      model_class = registers.model[model_name]
      hp = model_class.params(model_conf)
      model = hp.instantiate()

      tin = np.ones(hp.dims, dtype='float32') 
      tin = tin[np.newaxis, ...]
      tout = model(tin)
      self.assertEqual(tout.shape[1:], hp.z_dim)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
