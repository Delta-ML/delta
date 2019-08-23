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
''' loss implementation function '''
import tensorflow as tf

from delta import utils


#pylint: disable=too-many-arguments
def cross_entropy(logits,
                  labels,
                  input_length=None,
                  label_length=None,
                  smoothing=0.0,
                  reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
  '''
  cross entropy function for classfication and seq classfication
  :param, label_length, for seq task, this for target seq length, e.g. a b c </s>, 4
  '''
  del input_length

  onehot_labels = tf.cond(
      pred=tf.equal(tf.rank(logits) - tf.rank(labels), 1),
      true_fn=lambda: tf.one_hot(labels, tf.shape(logits)[-1], dtype=tf.int32),
      false_fn=lambda: labels)

  if label_length is not None:
    weights = utils.len_to_mask(label_length)
  else:
    weights = 1.0

  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels,
      logits=logits,
      weights=weights,
      label_smoothing=smoothing,
      reduction=reduction)

  return loss


def ctc_lambda_loss(logits, labels, input_length, label_length, blank_index=0):
  '''
  ctc loss function
  psram: logits, (B, T, D)
  psram: input_length,  (B, 1), input length of encoder
  psram: labels, (B, T)
  psram: label_length,  (B, 1), label length for convert dense label to sparse
  returns: loss, scalar
  '''
  ilen = tf.cond(
      pred=tf.equal(tf.rank(input_length), 1),
      true_fn=lambda: input_length,
      false_fn=lambda: tf.squeeze(input_length),
  )
  ilen = tf.cast(ilen, tf.int32)

  olen = tf.cond(
      pred=tf.equal(tf.rank(label_length), 1),
      true_fn=lambda: label_length,
      false_fn=lambda: tf.squeeze(label_length))
  olen = tf.cast(olen, tf.int32)

  deps = [
      tf.assert_rank(labels, 2, name='label_rank_check'),
      tf.assert_rank(logits, 3, name='logits_rank_check'),
      tf.assert_rank(ilen, 1, name='src_len_rank_check'),  # input_length
      tf.assert_rank(olen, 1, name='tgt_len_rank_check'),  # output_length
  ]

  labels, logits = ctc_data_transform(labels, logits, blank_index)

  with tf.control_dependencies(deps):
    # (B, 1)
    # blank index is consistent with Espnet, zero
    batch_loss = tf.nn.ctc_loss(
        labels=labels,
        inputs=logits,
        sequence_length=ilen,
        time_major=False,
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
        ignore_longer_outputs_than_inputs=False)
  return batch_loss

def ctc_data_transform(labels, logits, blank_index):
  '''
  data transform according blank_index
  '''
  if blank_index < 0 or blank_index is None:
    raise ValueError('blank_index must be greater than or equal to zero')

  num_class = logits.shape[2] - 1
  if blank_index != num_class:
    logits = tf.concat([logits[:, :, :blank_index],
                        logits[:, :, blank_index + 1:],
                        logits[:, :, blank_index:blank_index + 1]
                       ], axis=2)

  labels = tf.cast(labels, tf.int32)
  labels_idx = tf.where(tf.not_equal(labels, 0))
  labels_values = tf.gather_nd(labels, labels_idx)
  labels_num_class = tf.zeros_like(labels_values, dtype=tf.int32) + num_class
  labels_values_change_blank = tf.where(tf.equal(labels_values, blank_index),
                                        labels_num_class,
                                        labels_values)
  labels_values = tf.where(labels_values_change_blank < blank_index,
                           labels_values_change_blank,
                           labels_values_change_blank - 1)
  labels_shape = tf.cast(tf.shape(labels), dtype=tf.int64)
  labels_sparse = tf.SparseTensor(indices=labels_idx,
                                  values=labels_values,
                                  dense_shape=labels_shape)

  return labels_sparse, logits

def crf_log_likelihood(tags_scores, labels, input_length, transitions):
  '''
  :param tags_scores:  [batch_size, max_seq_len, num_tags]
  :param labels:  [batch_size, max_seq_len]
  :param input_length:  [batch_size,]
  :param transitions: [num_tags, num_tags]
  :return: loss, transition_params
  '''
  log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
      inputs=tags_scores,
      tag_indices=labels,
      sequence_lengths=input_length,
      transition_params=transitions)

  loss = tf.reduce_mean(-log_likelihood)

  return loss, transition_params


def mask_sequence_loss(logits,
                       labels,
                       input_length,
                       label_length,
                       smoothing=0.0):
  '''
  softmax cross entropy loss for sequence to sequence
  :param logits: [batch_size, max_seq_len, vocab_size]
  :param labels: [batch_size, max_seq_len]
  :param input_length: [batch_size]
  :param label_length: [batch_size]
  :return: loss, scalar
  '''
  del smoothing
  del input_length

  if label_length is not None:
    weights = tf.cast(utils.len_to_mask(label_length), tf.float32)
  else:
    weights = tf.ones_like(labels)
  loss = tf.contrib.seq2seq.sequence_loss(logits, labels, weights)
  return loss
