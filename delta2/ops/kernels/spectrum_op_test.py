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
import delta.compat as tf
from absl import logging

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib
from delta import PACKAGE_ROOT_DIR


class SpecOpTest(tf.test.TestCase):
  ''' spectrum op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_spectrum(self):
    ''' test spectrum op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)
      logging.info(
          f"input shape: {input_data.shape}, sample rate dtype: {sample_rate.dtype}"
      )
      self.assertEqual(sample_rate, 16000)

      output = py_x_ops.spectrum(input_data, sample_rate)

      #pylint: disable=bad-whitespace
      output_true = np.array(
          [[-16.863441, -16.910473, -17.077059, -16.371634, -16.845686],
           [-17.922068, -20.396345, -19.396944, -17.331493, -16.118851],
           [-17.017776, -17.551350, -20.332376, -17.403994, -16.617926],
           [-19.873854, -17.644503, -20.679525, -17.093716, -16.535091],
           [-17.074402, -17.295971, -16.896650, -15.995432, -16.560730]])
      #pylint: enable=bad-whitespace

      self.assertEqual(tf.rank(output).eval(), 2)
      logging.info('Shape of spectrum: {}'.format(output.shape))
      self.assertAllClose(output.eval()[4:9, 4:9], output_true)


if __name__ == '__main__':
  tf.test.main()
