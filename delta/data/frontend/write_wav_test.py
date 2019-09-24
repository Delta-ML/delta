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
from delta.data.frontend.write_wav import WriteWav

class WriteWavTest(tf.test.TestCase):

  def test_write_wav(self):
    wav_path = str(
      Path(os.environ['MAIN_ROOT']).joinpath('delta/layers/ops/data/sm1_cln.wav'))

    with self.session():
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)
      write_wav = WriteWav.params().instantiate()
      new_path = str(Path(os.environ['MAIN_ROOT']).joinpath('delta/layers/ops/data/sm1_cln_new.wav'))
      writewav_op = write_wav(new_path, input_data, sample_rate)
      self.session(writewav_op)
      test_data, test_sample_rate = read_wav(new_path)
      self.assertAllEqual(input_data.eval(), test_data.eval())
      self.assertAllEqual(sample_rate.eval(),test_sample_rate.eval())

if __name__ == '__main__':
  tf.test.main()
