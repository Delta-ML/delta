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
''' ctc utils unittest '''

import numpy as np
import delta.compat as tf

from delta.utils import ctc_utils


class CTCUtilTest(tf.test.TestCase):
  ''' ctc utils unittest'''

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

    self.labels = np.asarray([[1, 1, 1, 3], [1, 1, 1, 0]], dtype=np.int32)

  def tearDown(self):
    ''' tear down '''

  def test_transform_preprocess(self):
    ''' unit test case for the transform_preprocess interface '''
    with self.cached_session():

      with self.assertRaises(ValueError) as valueErr:
        labels = ctc_utils.transform_preprocess(
            labels=None, blank_index=None, num_class=None)
      the_exception = valueErr.exception
      self.assertEqual(
          str(the_exception),
          'blank_index must be greater than or equal to zero')

      with self.assertRaises(ValueError) as valueErr:
        labels = ctc_utils.transform_preprocess(
            labels=None, blank_index=-10, num_class=None)
      the_exception = valueErr.exception
      self.assertEqual(
          str(the_exception),
          'blank_index must be greater than or equal to zero')

      with self.assertRaises(ValueError) as valueErr:
        labels = ctc_utils.transform_preprocess(
            labels=None, blank_index=10, num_class=10)
      the_exception = valueErr.exception
      self.assertEqual(
          str(the_exception),
          'blank_index must be less than or equal to num_class - 1')

      labels = ctc_utils.transform_preprocess(
          labels=None, blank_index=0, num_class=10)
      self.assertIsNone(labels)

      labels = ctc_utils.transform_preprocess(
          labels=tf.constant(self.labels), blank_index=0, num_class=10)
      labels_values = np.asarray([1, 1, 1, 3, 1, 1, 1])
      labels_index = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1],
                                 [1, 2]])
      labels_shape = np.asarray([2, 4])
      self.assertAllEqual(labels.eval().values, labels_values)
      self.assertAllEqual(labels.eval().indices, labels_index)
      self.assertAllEqual(labels.eval().dense_shape, labels_shape)

  def test_logits_blankid_to_last(self):
    ''' unit test case for the logits_blankid_to_last interface '''
    with self.cached_session():

      with self.assertRaises(ValueError) as valueErr:
        logits = ctc_utils.logits_blankid_to_last(
            logits=tf.constant(self.logits), blank_index=10)
      the_exception = valueErr.exception
      self.assertEqual(
          str(the_exception),
          'blank_index must be less than or equal to num_class - 1')

      logits = ctc_utils.logits_blankid_to_last(
          logits=tf.constant(self.logits), blank_index=0)
      logits_transform = np.asarray(
          [[[0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553, 0.633766],
            [0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436, 0.111121],
            [0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688, 0.0357786],
            [0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533, 0.0663296],
            [0.196634, 0.123377, 0.50648837, 0.00903441, 0.00623107, 0.158235]],
           [[0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508, 0.30176],
            [0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549, 0.24082],
            [0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456, 0.230246],
            [0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345, 0.280884],
            [0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046, 0.423286]]],
          dtype=np.float32)
      self.assertAllClose(logits.eval(), logits)

  def test_labels_blankid_to_last(self):
    ''' unit test case for the labels_blankid_to_last interface '''
    with self.cached_session():

      with self.assertRaises(AssertionError) as assert_err:
        labels = ctc_utils.labels_blankid_to_last(
            labels=self.labels, blank_index=0, num_class=None)
      the_exception = assert_err.exception
      self.assertEqual(str(the_exception), 'The num_class should not be None!')

      labels = ctc_utils.labels_blankid_to_last(
          labels=tf.constant(self.labels), blank_index=0, num_class=6)
      labels_values = np.asarray([0, 0, 0, 2, 0, 0, 0])
      labels_index = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1],
                                 [1, 2]])
      labels_shape = np.asarray([2, 4])
      self.assertAllEqual(labels.eval().values, labels_values)
      self.assertAllEqual(labels.eval().indices, labels_index)
      self.assertAllEqual(labels.eval().dense_shape, labels_shape)

      labels = ctc_utils.labels_blankid_to_last(
          labels=tf.constant(self.labels), blank_index=2, num_class=6)
      labels_values = np.asarray([1, 1, 1, 2, 1, 1, 1])
      labels_index = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1],
                                 [1, 2]])
      labels_shape = np.asarray([2, 4])
      self.assertAllEqual(labels.eval().values, labels_values)
      self.assertAllEqual(labels.eval().indices, labels_index)
      self.assertAllEqual(labels.eval().dense_shape, labels_shape)

  def test_labels_last_to_blankid(self):
    ''' unit test case for the labels_last_to_blankid interface '''
    with self.cached_session():

      labels = ctc_utils.labels_last_to_blankid(
          labels=tf.constant(self.labels), blank_index=0, num_class=None)
      labels_values = np.asarray([2, 2, 2, 4, 2, 2, 2])
      labels_index = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1],
                                 [1, 2]])
      labels_shape = np.asarray([2, 4])
      self.assertAllEqual(labels.eval().values, labels_values)
      self.assertAllEqual(labels.eval().indices, labels_index)
      self.assertAllEqual(labels.eval().dense_shape, labels_shape)

      labels = ctc_utils.labels_last_to_blankid(
          labels=tf.constant(self.labels), blank_index=2, num_class=None)
      labels_values = np.asarray([1, 1, 1, 4, 1, 1, 1])
      labels_index = np.asarray([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1],
                                 [1, 2]])
      labels_shape = np.asarray([2, 4])
      self.assertAllEqual(labels.eval().values, labels_values)
      self.assertAllEqual(labels.eval().indices, labels_index)
      self.assertAllEqual(labels.eval().dense_shape, labels_shape)


if __name__ == '__main__':
  tf.test.main()
