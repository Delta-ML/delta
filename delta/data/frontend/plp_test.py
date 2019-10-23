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
from pathlib import Path
import numpy as np
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.plp import Plp
from delta import PACKAGE_ROOT_DIR


class PlpTest(tf.test.TestCase):

  def test_plp(self):
    wav_path = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)

      plp = Plp.params({
          'window_length': 0.025,
          'frame_length': 0.010,
          'plp_order': 12
      }).instantiate()
      plp_test = plp(input_data, sample_rate)

      output_true = np.array(
          [[-0.209490, -0.326126, 0.010536, -0.027167, -0.117118],
           [-0.020293, -0.454695, -0.104243, 0.001560, -0.234854],
           [-0.015118, -0.444044, -0.156695, -0.086221, -0.319310],
           [-0.031856, -0.130708, 0.047435, -0.089916, -0.160247],
           [0.052763, -0.271487, 0.011329, 0.025320, 0.012851]])

      self.assertEqual(tf.rank(plp_test).eval(), 2)
      self.assertAllClose(plp_test.eval()[50:55, 5:10], output_true, rtol=1e-05, atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
