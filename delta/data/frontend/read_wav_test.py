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
"""The model tests OP of read_wav """

import delta.compat as tf
from pathlib import Path
import librosa
from delta.data.frontend.read_wav import ReadWav
from core.ops import PACKAGE_OPS_DIR


class ReadWavTest(tf.test.TestCase):
  """
  ReadWav OP test.
  """

  def test_read_wav(self):
    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      config = {'speed': 1.0}
      read_wav = ReadWav.params(config).instantiate()
      audio_data, sample_rate = read_wav(wav_path)
      audio_data_true, sample_rate_true = librosa.load(wav_path, sr=16000)
      if (config['speed'] == 1.0):
        self.assertAllClose(audio_data.eval() / 32768, audio_data_true)
        self.assertAllClose(sample_rate.eval(), sample_rate_true)


if __name__ == '__main__':
  tf.test.main()
