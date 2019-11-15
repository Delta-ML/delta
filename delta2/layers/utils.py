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
"""Utilities for building model layers."""

import math
import delta.compat as tf
# pylint: disable=no-name-in-module
from tensorflow.python.keras import backend as K

import delta.utils as utils


# pylint: disable=invalid-name
def gelu(x):
  """An approximation of gelu.
     See: https://arxiv.org/pdf/1606.08415.pdf
  """
  return 0.5 * x * (
      1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


def cut_or_padding(origin_t, new_length, padding_token=0):
  """
  If too long, cut the tensor; else pad the tensor.
  origin_t: [batch_size, time_steps_1] or [time_steps_1]
  new_t: [batch_size, time_steps_2] or [time_steps_2]
  """

  if len(origin_t.get_shape()) == 1:
    dim = 1
    cur_length = tf.shape(origin_t)[0]
  elif len(origin_t.get_shape()) == 2:
    dim = 2
    cur_length = tf.shape(origin_t)[1]
  else:
    raise ValueError("origin_t should be a tensor with rank 1 or 2.")

  def cut_tensor():
    if dim == 1:
      new_t = origin_t[:new_length]
    else:
      new_t = origin_t[:, :new_length]
    return new_t

  def pad_tail_tensor():
    if dim == 1:
      shape = tf.constant([1, 2])
      indices = tf.constant([[0, 1]])
    else:
      shape = tf.constant([2, 2])
      indices = tf.constant([[1, 1]])
    updates = [new_length - cur_length]
    paddings = tf.scatter_nd(indices, updates, shape)
    new_t = tf.pad(
        origin_t, paddings, "CONSTANT", constant_values=padding_token)
    return new_t

  new_t = tf.cond(
      cur_length < new_length, true_fn=pad_tail_tensor, false_fn=cut_tensor)

  if dim == 1:
    new_t.set_shape([new_length])
  else:
    new_t.set_shape([origin_t.get_shape()[0], new_length])

  return new_t


def compute_sen_lens(inputs, padding_token=0):
  """
  Count how many words in a sentence.
  inputs: [..., time_steps]
  sen_lens: [...]
  """
  x_binary = tf.cast(tf.not_equal(inputs, padding_token), tf.int32)
  sen_lens = tf.reduce_sum(x_binary, axis=-1)
  ones = tf.ones_like(sen_lens)
  sen_lens = tf.where(tf.equal(sen_lens, utils.PAD_IDX), x=ones, y=sen_lens)
  return sen_lens


def compute_doc_lens(sen_lens):
  """
  Count how many sentences in a document.
  inputs: [..., time_steps]
  doc_lens: [...]
  """
  x_binary = tf.cast(tf.cast(sen_lens, tf.bool), tf.int32)
  doc_lens = tf.reduce_sum(x_binary, axis=-1)
  return doc_lens


def split_one_doc_to_true_len_sens(doc_t, split_token, padding_token,
                                   max_doc_len, max_sen_len):
  """
  Split a document to sentences with true sentence lengths.
  doc_t: [doc_word_len]
  out_t: [max_doc_len, max_sen_len]
  """
  if len(doc_t.get_shape()) == 1:
    split_token_index = tf.squeeze(
        tf.where(tf.equal(doc_t, split_token)), axis=1)
    split_token_index.set_shape([None])
    split_len_part_1 = split_token_index[:1] + 1
    split_len_part_2 = split_token_index[1:] - split_token_index[:-1]
    split_lens = tf.concat([split_len_part_1, split_len_part_2], axis=0)
    split_lens = cut_or_padding(
        split_lens, max_doc_len, padding_token=padding_token)
    new_doc_len = tf.reduce_sum(split_lens)
    splited_sentences = tf.split(doc_t[:new_doc_len], split_lens)
    splited_sentences = [
        cut_or_padding(s, max_sen_len) for s in splited_sentences
    ]
    out_t = tf.stack(splited_sentences)
    padding_tokens = tf.multiply(
        tf.ones_like(out_t, dtype=tf.int32), padding_token)
    out_t = tf.where(tf.equal(out_t, split_token), padding_tokens, out_t)
    return out_t

  raise ValueError("doc_t should be a tensor with rank 1.")


def get_pad_mask_from_token_idx(inputs, pad_idx):
  """
  get padding mask from the input token idx
  inputs: [batch_size, time_steps]
  mask: [batch_size, time_steps]
  """
  pad_mask = tf.cast(tf.math.greater(inputs, pad_idx), tf.int32)
  return pad_mask


def get_seg_mask_from_token_idx(inputs, seg_idx):
  """
  get padding mask from the input token idx
  inputs: [batch_size, time_steps]
  mask: [batch_size, time_steps]
  """
  seg_mask = tf.cast(tf.math.equal(inputs, seg_idx), tf.int32)
  return seg_mask
