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
"""The model tests spectrum FE."""

import os
import numpy as np
from pathlib import Path
import delta.compat as tf
from core.ops import PACKAGE_OPS_DIR
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.spectrum import Spectrum


class SpectrumTest(tf.test.TestCase):
  '''
  Spectum extraction test.
  '''

  def test_spectrum(self):
    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)

      spectrum = Spectrum.params({
          'window_length': 0.025,
          'snip_edges': True,
          'dither': 0.0
      }).instantiate()
      spectrum_test = spectrum(input_data, sample_rate)

      output_true = np.array(
          [[9.819611, 2.84503, 3.660894, 2.7779, 1.212233],
           [9.328745, 2.553949, 3.276319, 3.000918, 2.499342]])

      self.assertEqual(tf.rank(spectrum_test).eval(), 2)
      self.assertAllClose(
          spectrum_test.eval()[0:2, 0:5], output_true, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
