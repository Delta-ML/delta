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
import librosa

class ReadWavTest(tf.test.TestCase):

  def test_read_wav(self):
    wav_path = str(
      Path(os.environ['MAIN_ROOT']).joinpath('delta/layers/ops/data/sm1_cln.wav'))

    read_wav = ReadWav.params().instantiate()
    input_data, sample_rate = read_wav.call(wav_path)
    sess = tf.Session()
    audio_data = sess.run(input_data)
    sample_rate1 = sess.run(sample_rate)
    audio_data_true, sample_rate_true = librosa.load(wav_path, sr=16000)
    self.assertAllClose(audio_data, audio_data_true)
    self.assertAllClose(sample_rate1, sample_rate_true)

if __name__ == '__main__':
  tf.test.main()
