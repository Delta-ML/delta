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
"""The model tests Fbank FE."""

import os
import numpy as np
from pathlib import Path

import delta.compat as tf
from core.ops import PACKAGE_OPS_DIR
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.fbank import Fbank


class FbankTest(tf.test.TestCase):
  """
  Test Fbank FE using 8k/16k wav files.
  """

  def test_fbank(self):
    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)
      config = {
          'window_length': 0.025,
          'output_type': 1,
          'frame_length': 0.010,
          'snip_edges': True
      }
      fbank = Fbank.params(config).instantiate()
      fbank_test = fbank(input_data, sample_rate)

      self.assertEqual(tf.rank(fbank_test).eval(), 3)

      real_fank_feats = np.array(
          [[3.768338, 4.946218, 6.289874, 6.330853, 6.761764, 6.884573],
           [3.803553, 5.450971, 6.547878, 5.796172, 6.397846, 7.242926]])

      self.assertAllClose(
          np.squeeze(fbank_test.eval()[0:2, 0:6, 0]),
          real_fank_feats,
          rtol=1e-05,
          atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
