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
''' tensorflow metcis utils'''
import delta.compat as tf


def accuracy(logits, labels):
  ''' accuracy candies
  params:
    logits: [B, ..., D]
    labels: [B, ...]
  return:
    accuracy tensor
  '''
  with tf.name_scope('accuracy'):
    assert_rank = tf.assert_equal(tf.rank(logits), tf.rank(labels) + 1)
    assert_shape = tf.assert_equal(tf.shape(logits)[:-1], tf.shape(labels))
    with tf.control_dependencies([assert_rank, assert_shape]):
      predictions = tf.argmax(logits, axis=-1, output_type=tf.int64)
      labels = tf.cast(labels, tf.int64)
      return tf.reduce_mean(
          tf.cast(tf.equal(predictions, labels), dtype=tf.float32))


def confusion_matrix(logits, labels, num_class):
  ''' confusion matrix candies '''
  return tf.confusion_matrix(
      labels=tf.reshape(labels, [-1]),
      predictions=tf.reshape(tf.argmax(logits, -1), [-1]),
      num_classes=num_class)
