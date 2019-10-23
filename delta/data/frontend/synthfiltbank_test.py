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
from delta.data.frontend.analyfiltbank import Analyfiltbank
from delta.data.frontend.synthfiltbank import Synthfiltbank
from delta import PACKAGE_ROOT_DIR


class Test(tf.test.TestCase):

  def test_synthfiltbank(self):
    wav_path = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

    with self.cached_session(use_gpu=False, force_gpu=False):

      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)

      analyfiltbank = Analyfiltbank.params().instantiate()
      power_spc, phase_spc = analyfiltbank(input_data.eval(),
                                           sample_rate.eval())

      synthfiltbank = Synthfiltbank.params().instantiate()
      audio_data = synthfiltbank(power_spc, phase_spc, sample_rate.eval())

      self.assertAllClose(
          audio_data.eval().flatten()[500:550],
          input_data.eval().flatten()[500:550],
          rtol=1e-4,
          atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
