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
from delta.data.frontend.spectrum import Spectrum
import librosa

class SpectrumTest(tf.test.TestCase):

  def test_spectrum(self):
    wav_path = str(
        Path(os.environ['MAIN_ROOT']).joinpath('delta/layers/ops/data/sm1_cln.wav'))
    read_wav = ReadWav.params().instantiate()
    input_data, sample_rate = read_wav.call(wav_path)
    spectrum = Spectrum.params().instantiate()
    spectrum_test = spectrum(input_data, sample_rate)

    sess = tf.compat.v1.Session()
    spectrum_test1 = sess.run(spectrum_test)
    audio_data_true, sample_rate_true = librosa.load(wav_path, sr=16000)
    spectrum_true = librosa.stft(audio_data_true, n_fft=512, hop_length=160, win_length=400, window='hamm')
    print(spectrum_true.shape)
    # Because our algorithm pre-emphasized the audio_data, two spectrums are different.
    self.assertAllClose(spectrum_test1.transpose()[:, 0:10], spectrum_true[:, 0:10])

if __name__ == '__main__':
  tf.test.main()
