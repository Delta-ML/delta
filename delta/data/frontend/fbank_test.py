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
''' Fbank test'''

import tensorflow as tf

from delta.data.frontend.fbank import FBank

class FBankTest(tf.test.TestCase):
  ''' HParams unittest '''

  def test_fbank(self):
   '''test fbank'''
   time = 2
   sr = 16000
   channel = 1
   fbank = FBank.params().instantiate()
   inputs = tf.random.uniform([time*sr, channel], minval= -1, maxval=1, dtype=tf.dtypes.float32)
   x = fbank.call(inputs)
   shape = [198, 40]
   self.assertAllEqual(tf.shape(x), shape)

if __name__ == '__main__':
  tf.test.main()
