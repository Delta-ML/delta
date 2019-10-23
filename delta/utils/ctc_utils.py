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
''' global ASR CTC utils '''

import delta.compat as tf

def transform_preprocess(labels=None, blank_index=None, num_class=None):
  ''' Ensure that the value of blank_index is in a reasonable range,
      and transform the DenseTensor labels to a SparseTensor '''
  if blank_index is None or blank_index < 0:
    raise ValueError('blank_index must be greater than or equal to zero')

  if not num_class is None and blank_index > (num_class - 1):
    raise ValueError('blank_index must be less than or equal to num_class - 1')

  if labels is None:
    return None

  if not isinstance(labels, tf.SparseTensor):
    labels = tf.cast(labels, tf.int32)
    labels_idx = tf.where(tf.not_equal(labels, 0))
    labels_values = tf.gather_nd(labels, labels_idx)
    labels_shape = tf.cast(tf.shape(labels), dtype=tf.int64)
    labels = tf.SparseTensor(
        indices=labels_idx, values=labels_values, dense_shape=labels_shape)

  return labels


def logits_blankid_to_last(logits, blank_index):
  ''' Moves the blank_label cloumn to the end of the logit matrix '''
  num_class = logits.shape[2]
  transform_preprocess(blank_index=blank_index, num_class=num_class)

  if blank_index != (num_class - 1):
    logits = tf.concat([
        logits[:, :, :blank_index], logits[:, :, blank_index + 1:],
        logits[:, :, blank_index:blank_index + 1]
    ],
                       axis=2)

  return logits


def labels_blankid_to_last(labels, blank_index, num_class=None):
  ''' Change the value of blank_label elements from blank_index to num_class - 1'''
  assert num_class is not None, 'The num_class should not be None!'

  labels = transform_preprocess(
      labels=labels, blank_index=blank_index, num_class=num_class)

  labels_values = labels.values
  labels_num_class = tf.zeros_like(labels_values, dtype=tf.int32) + num_class
  labels_values_change_blank = tf.where(
      tf.equal(labels_values, blank_index), labels_num_class, labels_values)
  labels_values = tf.where(labels_values_change_blank < blank_index,
                           labels_values_change_blank,
                           labels_values_change_blank - 1)

  labels = tf.SparseTensor(
      indices=labels.indices,
      values=labels_values,
      dense_shape=labels.dense_shape)
  return labels


def labels_last_to_blankid(labels, blank_index, num_class=None):
  ''' Change the value of blank_label elements from num_classes - 1 to blank_index,
      after removing blank_index by decoder. '''
  labels = transform_preprocess(
      labels=labels, blank_index=blank_index, num_class=num_class)

  labels_values = labels.values
  labels_change_blank_id = tf.where(labels_values >= blank_index,
                                    labels_values + 1, labels_values)

  labels = tf.SparseTensor(
      indices=labels.indices,
      values=labels_change_blank_id,
      dense_shape=labels.dense_shape)

  return labels
