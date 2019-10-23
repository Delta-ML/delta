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

import delta.compat as tf
import os
import numpy as np
from pathlib import Path
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.spectrum import Spectrum
from delta import PACKAGE_ROOT_DIR


class SpectrumTest(tf.test.TestCase):

  def test_spectrum(self):
    wav_path = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)

      spectrum = Spectrum.params({'window_length': 0.025}).instantiate()
      spectrum_test = spectrum(input_data, sample_rate)

      output_true = np.array(
          [[-16.863441, -16.910473, -17.077059, -16.371634, -16.845686],
           [-17.922068, -20.396345, -19.396944, -17.331493, -16.118851],
           [-17.017776, -17.551350, -20.332376, -17.403994, -16.617926],
           [-19.873854, -17.644503, -20.679525, -17.093716, -16.535091],
           [-17.074402, -17.295971, -16.896650, -15.995432, -16.560730]])

      self.assertEqual(tf.rank(spectrum_test).eval(), 2)
      self.assertAllClose(spectrum_test.eval()[4:9, 4:9], output_true)


if __name__ == '__main__':
  tf.test.main()
