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
''' synthesis filter bank Op unit-test '''

import os
from pathlib import Path

import delta.compat as tf
from absl import logging

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib
from delta import PACKAGE_ROOT_DIR


class SfbOpTest(tf.test.TestCase):
  ''' synthesis filter bank op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_sfb(self):
    ''' test sfb op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      power_spc, phase_spc = py_x_ops.analyfiltbank(input_data, sample_rate)

      logging.info('Shape of power_spc: {}'.format(power_spc.eval().shape))
      logging.info('Shape of phase_spc: {}'.format(phase_spc.eval().shape))

      output = py_x_ops.synthfiltbank(power_spc.eval(), phase_spc.eval(),
                                      sample_rate)

      self.assertEqual(tf.rank(output).eval(), 1)
      logging.info('Shape of recovered signal: {}'.format(output.eval().shape))

      # beginning 400 samples are different, due to the overlap and add
      self.assertAllClose(
          output.eval().flatten()[500:550],
          input_data[500:550],
          rtol=1e-4,
          atol=1e-4)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
