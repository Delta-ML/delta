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
''' frame power  Op unit-test '''

import os
from pathlib import Path

import numpy as np
import delta.compat as tf
from absl import logging

from delta.data import feat as feat_lib
from delta.layers.ops import py_x_ops
from delta import PACKAGE_ROOT_DIR


class FrmPowOpTest(tf.test.TestCase):
  ''' frame_power op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_frmpow(self):
    ''' test frame_power op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.frame_pow(input_data, sample_rate)

      output_true = np.array([
          0.000018, 0.000011, 0.000010, 0.000010, 0.000010, 0.000010, 0.000008,
          0.000009, 0.000009, 0.000009, 0.000009, 0.000011, 0.090164, 0.133028,
          0.156547, 0.053551, 0.056670, 0.097706, 0.405659, 2.119505, 4.296845,
          6.139090, 6.623638, 6.136467, 7.595072, 7.904415, 7.655983, 6.771016,
          5.706427, 4.220942, 3.259599, 2.218259, 1.911394, 2.234246, 3.056905,
          2.534153, 0.464354, 0.013493, 0.021231, 0.148362, 0.364829, 0.627266,
          0.494912, 0.366029, 0.315408, 0.312441, 0.323796, 0.267505, 0.152856,
          0.045305
      ])
      self.assertEqual(tf.rank(output).eval(), 1)
      logging.info('Shape of frame_power: {}'.format(output.eval().shape))
      self.assertAllClose(output.eval().flatten()[:50], output_true)


if __name__ == '__main__':
  tf.test.main()
