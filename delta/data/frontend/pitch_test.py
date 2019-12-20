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
"""The model tests pitch FE."""

import delta.compat as tf
import os
from pathlib import Path
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.pitch import Pitch
import numpy as np
from core.ops import PACKAGE_OPS_DIR


class SpectrumTest(tf.test.TestCase):
  """
  Pitch extraction test.
  """

  def test_spectrum(self):
    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)

      pitch = Pitch.params({
          'window_length': 0.025,
          'soft_min_f0': 10.0
      }).instantiate()
      pitch_test = pitch(input_data, sample_rate)

      self.assertEqual(tf.rank(pitch_test).eval(), 2)

      output_true = [[-0.1366025, 143.8855], [-0.0226383, 143.8855],
                     [-0.08464742, 143.8855], [-0.08458386, 143.8855],
                     [-0.1208689, 143.8855]]

      self.assertAllClose(
          pitch_test.eval()[0:5, :], output_true, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
