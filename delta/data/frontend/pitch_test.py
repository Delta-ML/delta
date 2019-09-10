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

import tensorflow as tf
import os
from pathlib import Path
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.pitch import Pitch
import numpy as np

class PitchTest(tf.test.TestCase):

  def test_pitch(self):

    wav_path = str(
      Path(os.environ['MAIN_ROOT']).joinpath('delta/layers/ops/data/sm1_cln.wav'))
    with self.session():
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav.call(wav_path)
      pitch = Pitch.params().instantiate()
      pitch_test = pitch.call(input_data, sample_rate)

      output_true = np.array([
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        122.823532, 117.647057, 116.788322, 116.788322, 119.402985,
        119.402985, 119.402985, 119.402985, 119.402985, 123.076920,
        124.031006, 125.000000, 132.065216, 139.130432, 139.130432,
        137.931030, 126.108368, 114.285713, 115.107910, 122.070084,
        129.032257, 130.081299, 130.081299, 129.032257, 130.081299,
        131.147537, 129.032257, 125.000000, 120.300751, 115.107910
      ])

      self.assertAllClose(pitch_test.eval().flatten()[:50], output_true)

if __name__ == '__main__':
  tf.test.main()
