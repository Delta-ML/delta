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
''' plp op unit-test '''
import os
from pathlib import Path

import numpy as np
import delta.compat as tf
from absl import logging

from delta.data import feat as feat_lib
from delta.layers.ops import py_x_ops
from delta import PACKAGE_ROOT_DIR


class PLPOpTest(tf.test.TestCase):
  ''' plp op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_plp(self):
    ''' test plp op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.plp(input_data, sample_rate)

      #pylint: disable=bad-whitespace
      output_true = np.array(
          [[-0.209490, -0.326126, 0.010536, -0.027167, -0.117118],
           [-0.020293, -0.454695, -0.104243, 0.001560, -0.234854],
           [-0.015118, -0.444044, -0.156695, -0.086221, -0.319310],
           [-0.031856, -0.130708, 0.047435, -0.089916, -0.160247],
           [0.052763, -0.271487, 0.011329, 0.025320, 0.012851]])
      #pylint: enable=bad-whitespace

      self.assertEqual(tf.rank(output).eval(), 2)
      logging.info('Shape of PLP: {}'.format(output.shape))
      self.assertAllClose(output.eval()[50:55, 5:10], output_true, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
