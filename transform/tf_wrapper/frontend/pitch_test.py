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
from pathlib import Path
from transform.tf_wrapper.frontend.read_wav import ReadWav
from transform.tf_wrapper.frontend.pitch import Pitch
import numpy as np
from transform.tf_wrapper.ops import PACKAGE_OPS_DIR


class PitchTest(tf.test.TestCase):
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

      output_true = np.array(
        [
          [0.03881124, 0.3000031, - 0.02324523],
          [0.006756478, 0.3000097, 0.01047742],
          [0.02455365, 0.3000154, 0.00695902],
          [0.02453586, 0.3000221, 0.008448198],
          [0.03455311, 0.3000307, - 0.07547269],
          [0.04293294, 0.3000422, - 0.04193667]
        ]
      )

      # self.assertAllClose(
      #     pitch_test.eval()[0:6, :], output_true, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
