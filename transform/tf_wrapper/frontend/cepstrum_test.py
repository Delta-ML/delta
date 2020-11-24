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
"""The model tests Cepstrum FE."""

import numpy as np
from pathlib import Path

import delta.compat as tf
from transform.tf_wrapper.ops import PACKAGE_OPS_DIR
from transform.tf_wrapper.frontend.read_wav import ReadWav
from transform.tf_wrapper.frontend.cepstrum import Cepstrum


class CepstrumTest(tf.test.TestCase):
  """
  Cepstrum extraction test.
  """

  def test_cepstrum(self):

    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav.call(wav_path)
      input_data = input_data / 32768
      cepstrum = Cepstrum.params({'window_length': 0.025}).instantiate()
      cepstrum_test = cepstrum(input_data, sample_rate)

      output_true = np.array(
          [[0.525808, 0.579537, 0.159656, 0.014726, -0.1866810],
           [0.225988, 1.557304, 3.381828, 0.132935, 0.7128600],
           [-1.832759, -1.045178, 0.753158, 0.116107, -0.9307780],
           [-0.696277, 1.333355, 1.590942, 2.041829, -0.0805630],
           [-0.377375, 2.984320, 0.036302, 3.676640, 1.1709290]])

      # self.assertAllClose(cepstrum_test.eval()[15:20, 7:12], output_true)


if __name__ == '__main__':
  tf.test.main()
