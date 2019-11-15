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
from delta.data.frontend.read_wav import ReadWav
from delta.data.frontend.fbank import Fbank
from delta import PACKAGE_ROOT_DIR


class FbankTest(tf.test.TestCase):

  def test_fbank(self):
    wav_path = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)
      config = {'window_length': 0.025, 'output_type': 1, 'frame_length': 0.010}
      fbank = Fbank.params(config).instantiate()
      fbank_test = fbank(input_data, sample_rate)

      self.assertEqual(tf.rank(fbank_test).eval(), 3)


if __name__ == '__main__':
  tf.test.main()
