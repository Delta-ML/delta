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
''' read wav test'''
import tempfile
import numpy as np
from scipy.io import wavfile 

import tensorflow as tf

from delta.data.frontend.read_wav import Readwav 

class ReadwavTest(tf.test.TestCase):
  ''' HParams unittest '''

  def test_read_wav(self):
    '''test read wav'''

    time = 2
    sr = 16000
    def write_wav_file():
      _, wavefile = tempfile.mkstemp('.wav')
      data = np.random.randint(-32768, 32767, time*sr, dtype=np.int16)
      wavfile.write(wavefile, sr, data)
      return wavefile

    wavefile = write_wav_file()
    read_wav = Readwav.params().instantiate()
    waveforms = read_wav.call(wavefile)
    
    shape = [time*sr]
    self.assertAllEqual(tf.shape(waveforms), shape)

if __name__ == '__main__':
  tf.test.main()
