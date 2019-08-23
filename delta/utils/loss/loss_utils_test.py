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
''' loss implementation function unittest '''
import numpy as np
import tensorflow as tf
from absl import logging

from delta.utils.loss import loss_utils


class LossUtilTest(tf.test.TestCase):
  ''' loss util unittest'''

  def setUp(self):
    ''' setup '''
    # classfication: shape [2, 6]
    self.logits = np.array([[10, 23, 43, 23, 12, 23], [32, 10, 23, 45, 23, 0]],
                           dtype=np.float32)
    self.labels = np.array([2, 3], dtype=np.int32)
    # seq2seq: shape [2, 3, 6]
    self.seq_logits = np.array(
        [[[10, 2, 11, 23, 12, 42], [12, 32, 11, 2, 0, 0], [12, 32, 11, 2, 0, 0]
         ], [[3, 11, 2, 32, 4, 8], [12, 1, 32, 0, 0, 0], [0, 0, 0, 0, 0, 0]]],
        dtype=np.float32)
    self.seq_labels = np.array([[5, 1, 1], [3, 2, 0]], dtype=np.int32)
    self.input_length = np.array([[3, 2]], dtype=np.int32)
    self.label_length = np.array([3, 2], dtype=np.int32)

  def tearDown(self):
    ''' tear down '''

  def test_cross_entropy(self):
    ''' test cross entropy'''
    with self.cached_session():
      loss = loss_utils.cross_entropy(
          logits=tf.constant(self.logits),
          input_length=None,
          labels=tf.constant(self.labels),
          label_length=None)
      self.assertAllClose(loss.eval(), 0.0, rtol=1e-06, atol=1.5e-6)

      loss = loss_utils.cross_entropy(
          logits=tf.constant(self.seq_logits),
          input_length=tf.constant(self.input_length),
          labels=tf.constant(self.seq_labels),
          label_length=tf.constant(self.label_length),
          reduction=tf.losses.Reduction.NONE)
      self.assertEqual(loss.eval().shape, (2, 3))
      self.assertAllClose(
          loss.eval(),
          np.zeros((2, 3), dtype=np.float32),
          rtol=1e-06,
          atol=1.5e-6)

      loss = loss_utils.cross_entropy(
          logits=tf.constant(self.seq_logits),
          input_length=tf.constant(self.input_length),
          labels=tf.constant(self.seq_labels),
          label_length=tf.constant(self.label_length),
          reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      self.assertEqual(loss.eval().shape, ())
      self.assertAllClose(loss.eval(), 0.0, rtol=1e-06, atol=1.5e-6)

  def test_ctc_lambda_loss(self):
    ''' test ctc loss '''
    with self.cached_session():
      label_lens = np.expand_dims(np.asarray([5, 4]), 1)
      input_lens = np.expand_dims(np.asarray([5, 5]), 1)  # number of timesteps
      loss_log_probs = [9.409339, 4.275597]

      # dimensions are batch x time x categories
      labels = np.asarray([[1, 2, 5, 4, 5], [0, 1, 2, 0, 0]])
      inputs = np.asarray(
          [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
            [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
            [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
            [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
            [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
           [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
            [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
            [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
            [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
            [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]],
          dtype=np.float32)

      loss = loss_utils.ctc_lambda_loss(
          labels=tf.constant(labels),
          logits=tf.constant(inputs),
          input_length=tf.constant(input_lens),
          label_length=tf.constant(label_lens),
          blank_index=0)
      self.assertEqual(loss.eval().shape[0], inputs.shape[0])
      self.assertAllClose(loss.eval(), loss_log_probs, atol=1e-05)
      self.assertAllClose(
          np.mean(loss.eval()), np.mean(loss_log_probs), atol=1e-05)

      # test when batch_size = 1, that is, one sample only
      ref = [9.409339]
      input_lens = np.asarray([5])
      label_lens = np.asarray([5])

      labels = np.asarray([[1, 2, 5, 4, 5]])
      inputs = np.asarray(
          [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
            [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
            [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
            [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
            [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]]
          ],
          dtype=np.float32)
      loss = loss_utils.ctc_lambda_loss(
          labels=tf.constant(labels),
          logits=tf.constant(inputs),
          input_length=tf.constant(input_lens),
          label_length=tf.constant(label_lens),
          blank_index=0)
      self.assertAllClose(loss.eval(), ref, atol=1e-05)
      self.assertAllClose(np.mean(loss.eval()), np.mean(ref), atol=1e-05)

  def test_ctc_data_transform(self):
    ''' test ctc_data_transform '''
    with self.cached_session():
      inputs = np.asarray(
            [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
              [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
              [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688]]],
            dtype=np.float32)
      labels = np.asarray([[1, 2, 3, 4, 1], [2, 1, 1, 1, 1]]) 
    
      blank_index = 0
      labels_after_transform, inputs_after_transform = loss_utils.ctc_data_transform(labels, inputs, blank_index)
      labels_after_transform = tf.sparse_tensor_to_dense(labels_after_transform) 
      new_labels = [[0, 1, 2, 3, 0], [1, 0, 0, 0, 0]]
      new_inputs = [[[0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553, 0.633766],
                     [0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436, 0.111121],
                     [0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688, 0.0357786]]]
      self.assertAllEqual(labels_after_transform, new_labels)
      self.assertAllClose(inputs_after_transform, new_inputs)

      blank_index = 2
      labels_after_transform, inputs_after_transform = loss_utils.ctc_data_transform(labels, inputs, blank_index)
      labels_after_transform = tf.sparse_tensor_to_dense(labels_after_transform)
      new_labels = [[1, 4, 2, 3, 1], [4, 1, 1, 1, 1]]
      new_inputs = [[[0.633766, 0.221185, 0.0129757, 0.0142857, 0.0260553, 0.0917319],
                     [0.111121, 0.588392, 0.0055756, 0.00569609, 0.010436, 0.278779],
                     [0.0357786, 0.633813, 0.00249248, 0.00272882, 0.0037688, 0.321418]]] 
      self.assertAllClose(labels_after_transform, new_labels)
      self.assertAllClose(inputs_after_transform, new_inputs)
 
      blank_index = 5 
      labels_after_transform, inputs_after_transform = loss_utils.ctc_data_transform(labels, inputs, blank_index)
      labels_after_transform = tf.sparse_tensor_to_dense(labels_after_transform)
      new_labels = [[1, 2, 3, 4, 1], [2, 1, 1, 1, 1]]
      new_inputs = [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
                     [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
                     [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688]]]  
      self.assertAllClose(labels_after_transform, new_labels)
      self.assertAllClose(inputs_after_transform, new_inputs)

  def test_crf_loss(self):
    ''' test crf loss '''
    with self.cached_session():
      loss_true = np.float32(5.5096426)
      logits = np.asarray([[[0.3, 0.4, 0.3], [0.1, 0.9, 0.0], [0.2, 0.7, 0.1],
                            [0.3, 0.2, 0.5], [0.6, 0.2, 0.2]]],
                          dtype=np.float32)  # [1,5,3]
      trans_params = tf.fill([3, 3], 0.5, name='trans_params')
      labels = np.asarray([[0, 1, 2, 0, 1]])  # shape=[1,5]
      sequence_lengths = np.asarray([5])  # shape=[1,]
      loss, _ = loss_utils.crf_log_likelihood(
          tf.constant(logits), tf.constant(labels),
          tf.constant(sequence_lengths), trans_params)

      self.assertEqual(loss.eval(), loss_true)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
