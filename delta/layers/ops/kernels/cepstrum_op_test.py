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
''' cepstrum op unit-test '''
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib


class CepsOpTest(tf.test.TestCase):
  ''' cepstrum op unittest'''

  def setUp(self):
    '''set up'''
    self.wavpath = str(
        Path(os.environ['MAIN_ROOT']).joinpath(
            'delta/layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_cepstrum(self):
    ''' test cepstrum op'''
    with self.session():
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.cepstrum(input_data, sample_rate)

      output_true = np.array([
          -39.319466, -0.525144, 1.254634, -1.000523, 3.269833, 1.122467,
          0.374241, 2.331129, 1.194314, -2.226894, -0.080651, 0.341422,
          3.090863, -39.658028, -0.491055, 1.300456, -0.978587, 3.614898,
          1.891401, 0.044584, 0.297294, 1.899952, -1.267939, -0.179974,
          0.934334, 1.845452, -39.940613, -0.489544, 1.108610, 0.293507,
          3.229793, 0.759369, -0.030200, 1.158299, 2.234188, -1.235799,
          -1.212533, 0.813340, 3.036731, -39.713657, -0.061213, 1.318654,
          -1.193706, 2.612660, 0.184458, 0.791051, 2.485928, 2.912481,
          -0.641628, -1.924919
      ])
      self.assertEqual(tf.rank(output).eval(), 1)
      self.assertAllClose(output.eval().flatten()[:50], output_true)


if __name__ == '__main__':
  tf.test.main()
