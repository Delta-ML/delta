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
"""The model tests framepow FE."""

import os
import numpy as np
from pathlib import Path

import delta.compat as tf
from core.ops import PACKAGE_OPS_DIR
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.framepow import Framepow


class FramepowTest(tf.test.TestCase):
  """
  Framepow extraction test.
  """

  def test_framepow(self):
    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)

      framepow = Framepow.params({
          'window_length': 0.025,
          'frame_length': 0.010
      }).instantiate()
      framepow_test = framepow(input_data, sample_rate)

      real_framepow_feats = np.array(
          [9.819611, 9.328745, 9.247337, 9.26451, 9.266059])

      self.assertEqual(tf.rank(framepow_test).eval(), 1)
      self.assertAllClose(framepow_test.eval()[0:5], real_framepow_feats)


if __name__ == '__main__':
  tf.test.main()
