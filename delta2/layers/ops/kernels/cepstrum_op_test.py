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
import delta.compat as tf
from absl import logging

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib
from delta import PACKAGE_ROOT_DIR


class CepsOpTest(tf.test.TestCase):
  ''' cepstrum op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_cepstrum(self):
    ''' test cepstrum op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      output = py_x_ops.cepstrum(input_data, sample_rate)

      #pylint: disable=bad-whitespace
      output_true = np.array(
          [[0.525808, 0.579537, 0.159656, 0.014726, -0.1866810],
           [0.225988, 1.557304, 3.381828, 0.132935, 0.7128600],
           [-1.832759, -1.045178, 0.753158, 0.116107, -0.9307780],
           [-0.696277, 1.333355, 1.590942, 2.041829, -0.0805630],
           [-0.377375, 2.984320, 0.036302, 3.676640, 1.1709290]])
      #pylint: enable=bad-whitespace

      self.assertEqual(tf.rank(output).eval(), 2)
      logging.info('Shape of cepstrum: {}'.format(output.shape))
      self.assertAllClose(output.eval()[15:20, 7:12], output_true)


if __name__ == '__main__':
  tf.test.main()
