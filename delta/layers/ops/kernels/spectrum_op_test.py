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
''' spectrum Op unit-test '''
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib


class SpecOpTest(tf.test.TestCase):
  ''' spectrum op unittest'''

  def setUp(self):
    '''set up'''
    self.wavpath = str(
        Path(os.environ['MAIN_ROOT']).joinpath(
            'delta/layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_spectrum(self):
    ''' test spectrum op'''
    with self.session():
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.spectrum(input_data, sample_rate)

      output_true = np.array([
          -16.018925, -16.491777, -16.903442, -18.108875, -19.477205,
          -19.039738, -17.066263, -16.530647, -16.033670, -15.492795,
          -15.347169, -16.443783, -15.385968, -15.631793, -16.286760,
          -16.555447, -15.107640, -15.158586, -16.397518, -14.803325,
          -15.173873, -15.785010, -15.551179, -15.487743, -15.732930,
          -15.610220, -15.314099, -14.765355, -14.572725, -13.482535,
          -13.463938, -14.457010, -16.253452, -15.444997, -13.472414,
          -12.852523, -13.163157, -13.957175, -14.148843, -13.527264,
          -12.840333, -13.056757, -14.582790, -13.900843, -13.864534,
          -14.037180, -15.386706, -16.500109, -16.309618, -13.585808
      ])
      self.assertEqual(tf.rank(output).eval(), 1)
      self.assertAllClose(output.eval().flatten()[:50], output_true)


if __name__ == '__main__':
  tf.test.main()
