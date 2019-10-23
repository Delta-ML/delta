# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' ctc tensorflow decode unittest '''

import numpy as np
import delta.compat as tf

from delta.utils.decode import tf_ctc


class DecodeUtilTest(tf.test.TestCase):
  ''' ctc tensorflow decode util unittest'''

  def setUp(self):
    super().setUp()
    ''' setup '''
    self.logits = np.asarray(
        [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
          [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
          [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
          [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
          [0.158235, 0.196634, 0.123377, 0.50648837, 0.00903441, 0.00623107]],
         [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
          [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
          [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
          [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
          [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]],
        dtype=np.float32)

    self.sequence_lens = np.expand_dims(np.asarray([5, 5], dtype=np.int32), 1)
    self.decode_result = np.asarray([[1, 1, 1, 3], [1, 1, 1, 0]],
                                    dtype=np.int32)

  def tearDown(self):
    ''' tear down '''

  def test_ctc_decode_blankid_to_last(self):
    ''' unit test case for the ctc_decode_blankid_to_last interface '''
    with self.cached_session():
      logits, sequence_lens, blank_id = tf_ctc.ctc_decode_blankid_to_last(
          tf.constant(self.logits), tf.constant(self.sequence_lens))
      logits_after_transform = np.asarray(
          [[[0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553, 0.633766],
            [0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508, 0.30176]],
           [[0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436, 0.111121],
            [0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549, 0.24082]],
           [[0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688, 0.0357786],
            [0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456, 0.230246]],
           [[0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533, 0.0663296],
            [0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345, 0.280884]],
           [[0.196634, 0.123377, 0.5064884, 0.00903441, 0.00623107, 0.158235],
            [0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046, 0.423286]]],
          dtype=np.float32)
      sequence_lens_after_transform = np.asarray([5, 5], dtype=np.int32)
      blank_id_after_transform = 0
      self.assertAllClose(logits.eval(), logits_after_transform)
      self.assertAllEqual(sequence_lens.eval(), sequence_lens_after_transform)
      self.assertAllEqual(blank_id, blank_id_after_transform)

      logits, sequence_lens, blank_id = tf_ctc.ctc_decode_blankid_to_last(
          tf.constant(self.logits), tf.constant(self.sequence_lens), blank_id=2)
      logits_after_transform = np.asarray(
          [[[0.633766, 0.221185, 0.0129757, 0.0142857, 0.0260553, 0.0917319],
            [0.30176, 0.28562, 0.0862751, 0.0816851, 0.161508, 0.0831517]],
           [[0.111121, 0.588392, 0.0055756, 0.00569609, 0.010436, 0.278779],
            [0.24082, 0.397533, 0.0546814, 0.0557528, 0.19549, 0.0557226]],
           [[0.0357786, 0.633813, 0.00249248, 0.00272882, 0.0037688, 0.321418],
            [0.230246, 0.450868, 0.038309, 0.0391602, 0.202456, 0.0389607]],
           [[0.0663296, 0.643849, 0.00283995, 0.0035545, 0.00331533, 0.280111],
            [0.280884, 0.429522, 0.0339046, 0.0326856, 0.190345, 0.0326593]],
           [[0.158235, 0.196634, 0.5064884, 0.00903441, 0.00623107, 0.123377],
            [0.423286, 0.315517, 0.0393744, 0.0339315, 0.154046, 0.0338439]]],
          dtype=np.float32)
      sequence_lens_after_transform = np.asarray([5, 5], dtype=np.int32)
      blank_id_after_transform = 2
      self.assertAllClose(logits.eval(), logits_after_transform)
      self.assertAllEqual(sequence_lens.eval(), sequence_lens_after_transform)
      self.assertAllEqual(blank_id, blank_id_after_transform)

  def test_ctc_greedy_decode(self):
    ''' ctc tensorflow greedy decode unittest '''

    with self.cached_session():
      decode_result, _ = tf_ctc.ctc_greedy_decode(
          tf.constant(self.logits),
          tf.constant(self.sequence_lens),
          merge_repeated=True)
      self.assertAllEqual(decode_result.eval(), [[1, 3], [1, 0]])

      decode_result, _ = tf_ctc.ctc_greedy_decode(
          tf.constant(self.logits),
          tf.constant(self.sequence_lens),
          merge_repeated=False)
      self.assertAllEqual(decode_result.eval(), [[1, 1, 1, 3], [1, 1, 1, 0]])

  def test_ctc_beam_search_decode(self):
    ''' ctc tensorflow beam search unittest'''

    with self.cached_session():
      decode_result, _ = tf_ctc.ctc_beam_search_decode(
          tf.constant(self.logits),
          tf.constant(self.sequence_lens),
          beam_width=1,
          top_paths=1)
      self.assertAllEqual(decode_result[0].eval(), [[1], [1]])


if __name__ == '__main__':
  tf.test.main()
