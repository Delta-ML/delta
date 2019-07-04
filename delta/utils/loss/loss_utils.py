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
from tensorflow.keras import backend as K  #pylint: disable=import-error

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


def ctc_lambda_loss(logits, labels, input_length, label_length, smoothing=0.0):
  '''
  ctc loss function
  psram: logits, (B, T, D)
  psram: input_length,  (B, 1), input length of encoder
  psram: labels, (B, T)
  psram: label_length,  (B, 1), label length for convert dense label to sparse
  returns: loss, scalar
  '''
  del smoothing

  ilen = tf.cond(
      pred=tf.equal(tf.rank(input_length), 1),
      true_fn=lambda: tf.expand_dims(input_length, axis=-1),
      false_fn=lambda: input_length,
  )
  olen = tf.cond(
      pred=tf.equal(tf.rank(label_length), 1),
      true_fn=lambda: tf.expand_dims(label_length, axis=-1),
      false_fn=lambda: label_length,
  )
  deps = [
      tf.assert_rank(labels, 2),
      tf.assert_rank(logits, 3),
      tf.assert_rank(ilen, 2),  # input_length
      tf.assert_rank(olen, 2),  # output_length
  ]
  with tf.control_dependencies(deps):
    # (B, 1)
    batch_loss = K.ctc_batch_cost(labels, logits, ilen, olen)
    loss = tf.reduce_mean(batch_loss)

  return loss

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
