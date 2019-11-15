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
''' pitch op unit-test '''
import os
from pathlib import Path

import numpy as np
import delta.compat as tf
from absl import logging

from delta.data import feat as feat_lib
from delta.layers.ops import py_x_ops
from delta import PACKAGE_ROOT_DIR


class PitchOpTest(tf.test.TestCase):
  ''' pitch op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_pitch(self):
    ''' test pitch op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      # read wave
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.pitch(input_data, sample_rate)

      output_true = np.array([
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
          0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
          122.823532, 117.647057, 116.788322, 116.788322, 119.402985,
          119.402985, 119.402985, 119.402985, 119.402985, 123.076920,
          124.031006, 125.000000, 132.065216, 139.130432, 139.130432,
          137.931030, 126.108368, 114.285713, 115.107910, 122.070084,
          129.032257, 130.081299, 130.081299, 129.032257, 130.081299,
          131.147537, 129.032257, 125.000000, 120.300751, 115.107910
      ])
      self.assertEqual(tf.rank(output).eval(), 1)
      logging.info('Shape of pitch: {}'.format(output.eval().shape))
      self.assertAllClose(output.eval().flatten()[:50], output_true)


if __name__ == '__main__':
  tf.test.main()
