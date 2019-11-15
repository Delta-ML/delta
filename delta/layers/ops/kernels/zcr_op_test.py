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
''' zcr Op unit-test '''
import os
from pathlib import Path

import numpy as np
import delta.compat as tf
from absl import logging

from delta.data import feat as feat_lib
from delta.layers.ops import py_x_ops
from delta import PACKAGE_ROOT_DIR


class ZcrOpTest(tf.test.TestCase):
  ''' zero-cross-rate op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_zcr(self):
    ''' test zcr op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.zcr(input_data, sample_rate)

      output_true = np.array([
          0.406250, 0.418750, 0.425000, 0.407500, 0.393750, 0.392500, 0.388750,
          0.417500, 0.427500, 0.456250, 0.447500, 0.386250, 0.357500, 0.282500,
          0.232500, 0.262500, 0.282500, 0.295000, 0.220000, 0.157500, 0.125000,
          0.107500, 0.100000, 0.092500, 0.092500, 0.095000, 0.097500, 0.105000,
          0.100000, 0.112500, 0.120000, 0.132500, 0.130000, 0.135000, 0.112500,
          0.120000, 0.090000, 0.080000, 0.070000, 0.080000, 0.087500, 0.092500,
          0.097500, 0.097500, 0.112500, 0.090000, 0.065000, 0.087500, 0.175000,
          0.240000
      ])
      self.assertEqual(tf.rank(output).eval(), 1)
      logging.info('Shape of zero-cross-rate: {}'.format(output.eval().shape))
      self.assertAllClose(output.eval().flatten()[:50], output_true)


if __name__ == '__main__':
  tf.test.main()
