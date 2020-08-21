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
"""The model tests MelSpectrum FE."""


from pathlib import Path
import delta.compat as tf
from transform.tf_wrapper.ops import PACKAGE_OPS_DIR
from transform.tf_wrapper.frontend.read_wav import ReadWav
from transform.tf_wrapper.frontend.mel_spectrum import MelSpectrum

class MelSpectrumTest(tf.test.TestCase):
  """
  MelSpectrum extraction test.
  """
  def test_mel_spectrum(self):
    # 16kHz test
    wav_path_16k = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):
      # value test
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path_16k)
      print(sample_rate.eval())
      config = {'type': 'MelSpectrum', 'window_type': 'hann',
                'upper_frequency_limit': 7600, 'filterbank_channel_count': 80,
                'lower_frequency_limit': 80, 'dither': 0.0,
                'window_length': 0.025, 'frame_length': 0.010,
                'remove_dc_offset': False, 'preEph_coeff': 0.0,
                'output_type': 3, 'sample_rate': 16000}
      mel_spectrum = MelSpectrum.params(config).instantiate()
      mel_spectrum_test = mel_spectrum(input_data, sample_rate)
      print(mel_spectrum_test.eval()[0:2, 0:10])


if __name__ == '__main__':
  tf.test.main()



