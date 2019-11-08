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
from delta.data.frontend.mfcc import Mfcc
import numpy as np
from delta import PACKAGE_ROOT_DIR

class MfccTest(tf.test.TestCase):

  def test_mfcc(self):
    wav_path = str(
      Path(PACKAGE_ROOT_DIR).joinpath(
        'layers/ops/data/sm1_cln.wav'))

    with self.session():
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)
      # config = {}
      mfcc = Mfcc.params().instantiate()
      mfcc_test = mfcc(input_data, sample_rate)

      self.assertEqual(tf.rank(mfcc_test).eval(), 3)

      real_mfcc_feats = np.array(
        [[-30.58736, -7.088838, -10.67966, -1.646479, -4.36086],
         [-30.73371, -6.128432, -7.930599, 3.208357, -1.086456]])

      self.assertAllClose(np.squeeze(mfcc_test.eval()[0, 0:2, 1:6]), real_mfcc_feats, rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
  tf.test.main()



