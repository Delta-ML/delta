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
import tensorflow as tf

from delta.data import feat as feat_lib
from delta.layers.ops import py_x_ops


class PLPOpTest(tf.test.TestCase):
  ''' plp op unittest'''

  def setUp(self):
    '''set up'''
    self.wavpath = str(
        Path(os.environ['MAIN_ROOT']).joinpath(
            'delta/layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_plp(self):
    ''' test plp op'''
    with self.session():
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.plp(input_data, sample_rate)

      output_true = np.array([
          -0.000000, -0.959257, -0.095592, -0.219479, -0.104977, -0.185207,
          -0.153651, -0.081711, -0.156977, -0.072177, 0.077400, 0.027594,
          0.040156, -0.000000, -0.956464, -0.086729, -0.211084, -0.062403,
          -0.212304, -0.240348, -0.081032, -0.036527, -0.071906, 0.025969,
          0.004119, 0.003473, -0.000000, -0.952486, -0.094521, -0.143834,
          -0.133079, -0.244882, -0.175419, -0.040801, -0.071001, -0.134758,
          0.061415, 0.085666, 0.012909, -0.000000, -0.928211, -0.108592,
          -0.249340, -0.141225, -0.199109, -0.081247, -0.044329, -0.140386,
          -0.174557, -0.045552
      ])
      self.assertEqual(tf.rank(output).eval(), 1)
      self.assertAllClose(output.eval().flatten()[:50], output_true)


if __name__ == '__main__':
  tf.test.main()
