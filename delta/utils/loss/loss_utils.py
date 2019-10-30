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
import math
import delta.compat as tf
import tensorflow_addons as tfa

from delta import utils
from delta.utils import ctc_utils


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
  logits = ctc_utils.logits_blankid_to_last(
      logits=logits, blank_index=blank_index)

  num_class = logits.shape[2]
  labels = ctc_utils.labels_blankid_to_last(
      labels=labels, blank_index=blank_index, num_class=num_class)
  return labels, logits


def crf_log_likelihood(tags_scores, labels, input_length, transitions):
  '''
  :param tags_scores:  [batch_size, max_seq_len, num_tags]
  :param labels:  [batch_size, max_seq_len]
  :param input_length:  [batch_size,]
  :param transitions: [num_tags, num_tags]
  :return: loss, transition_params
  '''
  log_likelihood, transition_params = tfa.text.crf_log_likelihood(
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


#pylint: disable=too-many-locals
def arcface_loss(embedding,
                 labels,
                 out_num,
                 weights=None,
                 s=64.,
                 m=0.5,
                 limit_to_pi=True):
  '''
  https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py
  :param embedding: the input embedding vectors
  :param labels:  the input labels, the shape should be eg: (batch_size, 1)
  :param s: scalar value default is 64
  :param out_num: output class num
  :param weights: a tf.variable with shape (embedding.shape[-1], out_num)
                  or None to make a new one internally. default = None
  :param m: the margin value, default is 0.5
  :return: the final cacualted output, this output is send into the tf.nn.softmax directly
  '''
  cos_m = math.cos(m)
  sin_m = math.sin(m)
  mm = sin_m * m  # issue 1
  threshold = math.cos(math.pi - m)
  with tf.variable_scope('arcface_loss'):
    # inputs and weights norm
    embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
    embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
    if weights is None:
      weights = tf.get_variable(
          name='weights',
          shape=[embedding.shape[-1].value, out_num],
          initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    weights_norm = tf.norm(weights, axis=0, keep_dims=True)
    weights = tf.div(weights, weights_norm, name='norm_weights')
    # cos(theta+m)
    cos_t = tf.matmul(embedding, weights, name='cos_t')
    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(
        tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

    if limit_to_pi:
      # this condition controls the theta+m should in range [0, pi]
      #      0<=theta+m<=pi
      #     -m<=theta<=pi-m
      cond_v = cos_t - threshold
      cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

      keep_val = s * (cos_t - mm)
      cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    else:
      cos_mt_temp = cos_mt

    mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
    # mask = tf.squeeze(mask, 1)
    inv_mask = tf.subtract(1., mask, name='inverse_mask')

    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

    output = tf.add(
        tf.multiply(s_cos_t, inv_mask),
        tf.multiply(cos_mt_temp, mask),
        name='arcface_loss_output')
  return output


def focal_loss(logits, labels, alpha, gamma=2, name='focal_loss'):
  """
    Focal loss for multi classification
    :param logits: A float32 tensor of shape [batch_size num_class].
    :param labels: A int32 tensor of shape [batch_size, num_class] or [batch_size].
    :param alpha: A 1D float32 tensor for focal loss alpha hyper-parameter
    :param gamma: A scalar for focal loss gamma hyper-parameter.
    Returns: A tensor of the same shape as `lables`
    """
  if len(labels.shape) == 1:
    labels = tf.one_hot(labels, logits.shape[-1])
  else:
    labels = labels
  labels = tf.to_float(labels)

  y_pred = tf.nn.softmax(logits, dim=-1)
  L = -labels * tf.log(y_pred)
  L *= alpha * ((1 - y_pred)**gamma)
  loss = tf.reduce_sum(L)

  if tf.executing_eagerly():
    tf.contrib.summary.scalar(name, loss)
  else:
    tf.summary.scalar(name, loss)

  return loss
