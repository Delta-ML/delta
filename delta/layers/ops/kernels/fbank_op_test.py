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
''' fbank op unittest'''
import numpy as np
import delta.compat as tf

from delta.layers.ops import py_x_ops


class FbankOpTest(tf.test.TestCase):
  ''' fbank op unittest'''

  def setUp(self):
    super().setUp()
    ''' setup '''

  def tearDown(self):
    ''' tear donw '''

  def test_fbank(self):
    ''' test fbank op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      data = np.arange(513)
      spectrogram = tf.constant(data[None, None, :], dtype=tf.float32)
      sample_rate = tf.constant(22050, tf.int32)
      output = py_x_ops.fbank(
          spectrogram, sample_rate, filterbank_channel_count=20)

      output_true = np.array([
          1.887894, 2.2693727, 2.576507, 2.8156495, 3.036504, 3.2296343,
          3.4274294, 3.5987632, 3.771217, 3.937401, 4.0988584, 4.2570987,
          4.4110703, 4.563661, 4.7140336, 4.8626432, 5.009346, 5.1539173,
          5.2992935, 5.442024
      ])
      self.assertEqual(tf.rank(output).eval(), 3)
      self.assertEqual(output.shape, (1, 1, 20))
      self.assertAllClose(output.eval(), output_true[None, None, :])


if __name__ == '__main__':
  tf.test.main()
