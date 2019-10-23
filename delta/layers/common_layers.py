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
"""Common layers."""

import delta.compat as tf
from absl import logging

from delta.data.feat import speech_ops

#pylint: disable=invalid-name

def splice_layer(x, name, context):
  '''
  Splice a tensor along the last dimension with context.
  e.g.:
  t = [[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]
  splice_tensor(t, [0, 1]) =
      [[[1, 2, 3, 4, 5, 6],
        [4, 5, 6, 7, 8, 9],
        [7, 8, 9, 7, 8, 9]]]

  Args:
    tensor: a tf.Tensor with shape (B, T, D) a.k.a. (N, H, W)
    context: a list of context offsets

  Returns:
    spliced tensor with shape (..., D * len(context))
  '''
  with tf.variable_scope(name):
    input_shape = tf.shape(x)
    B, T = input_shape[0], input_shape[1]
    context_len = len(context)
    array = tf.TensorArray(x.dtype, size=context_len)
    for idx, offset in enumerate(context):
      begin = offset
      end = T + offset
      if begin < 0:
        begin = 0
        sliced = x[:, begin:end, :]
        tiled = tf.tile(x[:, 0:1, :], [1, abs(offset), 1])
        final = tf.concat((tiled, sliced), axis=1)
      else:
        end = T
        sliced = x[:, begin:end, :]
        tiled = tf.tile(x[:, -1:, :], [1, abs(offset), 1])
        final = tf.concat((sliced, tiled), axis=1)
      array = array.write(idx, final)
    spliced = array.stack()
    spliced = tf.transpose(spliced, (1, 2, 0, 3))
    spliced = tf.reshape(spliced, (B, T, -1))
  return spliced


#pylint: disable=too-many-arguments
def tdnn(x,
         name,
         in_dim,
         context,
         out_dim,
         has_bias=True,
         method='splice_layer'):
  '''
  TDNN implementation.

  Args:
    context:
      a int of left and right context, or
      a list of context indexes, e.g. (-2, 0, 2).
    method:
      splice_layer: use column-first patch-based copy.
      splice_op: use row-first while_loop copy.
      conv1d: use conv1d as TDNN equivalence.
  '''
  if hasattr(context, '__iter__'):
    context_size = len(context)
    if method in ('splice_op', 'conv1d'):
      msg = 'Method splice_op and conv1d does not support context list.'
      raise ValueError(msg)
    context_list = context
  else:
    context_size = context * 2 + 1
    context_list = range(-context, context + 1)
  with tf.variable_scope(name):
    if method == 'splice_layer':
      x = splice_layer(x, 'splice', context_list)
      x = linear(
          x, 'linear', [in_dim * context_size, out_dim], has_bias=has_bias)
    elif method == 'splice_op':
      x = speech_ops.splice(x, context, context)
      x = linear(
          x, 'linear', [in_dim * context_size, out_dim], has_bias=has_bias)
    elif method == 'conv1d':
      kernel = tf.get_variable(
          name='DW',
          shape=[context, in_dim, out_dim],
          dtype=tf.float32,
          initializer=tf.glorot_uniform_initializer())
      x = tf.nn.conv1d(x, kernel, stride=1, padding='SAME')
      if has_bias:
        b = tf.get_variable(
            name='bias',
            shape=[out_dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, b)
    else:
      raise ValueError('Unsupported method: %s.' % (method))
    return x


def conv2d(x, name, filter_size, in_channels, out_channels, strides, bias=True):
  """2D convolution."""
  with tf.variable_scope(name):
    kernel = tf.get_variable(
        name='DW',
        shape=[filter_size[0], filter_size[1], in_channels, out_channels],
        dtype=tf.float32,
        initializer=tf.initializers.glorot_uniform())
    if bias:
      b = tf.get_variable(
          name='bias',
          shape=[out_channels],
          dtype=tf.float32,
          initializer=tf.constant_initializer(0.0))
    out = tf.nn.conv2d(
        x, kernel, [1, strides[0], strides[1], 1], padding='SAME')
    if bias:
      out = tf.nn.bias_add(out, b)
    return out


def max_pool(x, ksize, strides):
  """Max Pooling."""
  return tf.nn.max_pool(
      x,
      ksize=[1, ksize[0], ksize[1], 1],
      strides=[1, strides[0], strides[1], 1],
      padding='VALID',
      name='max_pool')


def linear(x, names, shapes, has_bias=True):
  """Linear Layer."""
  assert len(shapes) == 2
  with tf.variable_scope(names):
    weights = tf.get_variable(
        name='weights',
        shape=shapes,
        initializer=tf.initializers.glorot_uniform())
    if has_bias:
      bias = tf.get_variable(
          name='bias',
          shape=shapes[1],
          initializer=tf.initializers.glorot_uniform())
      return tf.matmul(x, weights) + bias
    else:
      return tf.matmul(x, weights)


def attention(inputs, attention_size, time_major=False, return_alphas=False):
  """Attention layer."""
  if isinstance(inputs, tuple):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    inputs = tf.concat(inputs, 2)

  if time_major:
    # (T,B,D) => (B,T,D)
    inputs = tf.transpose(inputs, [1, 0, 2])

  time_size = inputs.shape[1].value  # T value - time size of the RNN layer
  hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

  # Trainable parameters
  W_omega = tf.get_variable(
      name='W_omega',
      initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
  b_omega = tf.get_variable(
      name='b_omega',
      initializer=tf.random_normal([attention_size], stddev=0.1))
  u_omega = tf.get_variable(
      name='u_omega',
      initializer=tf.random_normal([attention_size, 1], stddev=0.1))

  # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
  #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
  #v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
  #v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
  # (B, T, D) dot (D, Atten)

  logging.info('attention inputs: {}'.format(inputs.shape))
  inputs_reshaped = tf.reshape(inputs, [-1, hidden_size])
  dot = tf.matmul(inputs_reshaped, W_omega)
  dot = tf.reshape(dot, [-1, time_size, attention_size])
  v = tf.sigmoid(dot + b_omega)
  logging.info(f'attention vector: {v.shape}')
  # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
  # (B, T, Atten) dot (Atten)
  #vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
  v = tf.reshape(v, [-1, attention_size])
  vu = tf.matmul(v, u_omega)  # (B,T) shape
  vu = tf.squeeze(vu, axis=-1)
  vu = tf.reshape(vu, [-1, time_size])
  logging.info(f'attention energe: {vu.shape}')
  alphas = tf.nn.softmax(vu)  # (B,T) shape also

  # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
  # [batch, time] -> [batch, time, 1]
  alphas = tf.expand_dims(alphas, -1)
  # [batch, time, dim] -> [batch, dim]
  output = tf.reduce_sum(inputs * alphas, 1)

  if not return_alphas:
    return output

  return output, alphas


def embedding_look_up(text_inputs, vocab_size, embedding_size):
  """Embedding layer."""
  with tf.variable_scope("embedding"):
    W = tf.get_variable(
        name='W',
        initializer=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embedding_chars = tf.nn.embedding_lookup(W, text_inputs)
    embedding_chars_expanded = tf.expand_dims(embedding_chars, -1)
  return embedding_chars_expanded


#pylint: disable=too-many-locals
def conv_pool(embedded_chars_expanded, filter_sizes, embedding_size,
              num_filters, sequence_length):
  """
    text conv and max pooling to get one-dimension vector to representation of text
    :param filter_sizes:
    :return:
    """
  pooled_outputs = []
  for _, filter_size in enumerate(filter_sizes):
    with tf.variable_scope("conv-maxpool-%s" % filter_size):
      # Convolution Layer
      filter_shape = [filter_size, embedding_size, 1, num_filters]
      W = tf.get_variable(
          name='W', initializer=tf.truncated_normal(filter_shape, stddev=0.1))
      b = tf.get_variable(
          name='b', initializer=tf.constant(0.1, shape=[num_filters]))
      conv = tf.nn.conv2d(
          embedded_chars_expanded,
          W,
          strides=[1, 1, 1, 1],
          padding="VALID",
          name="conv")
      # Apply nonlinearity
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
      # Maxpooling over the outputs
      pooled = tf.nn.max_pool(
          h,
          ksize=[1, sequence_length - filter_size + 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding='VALID',
          name="pool")
      pooled_outputs.append(pooled)
  # Combine all the pooled features
  num_filters_total = num_filters * len(filter_sizes)

  h_pool = tf.concat(pooled_outputs, 3)

  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
  return h_pool_flat
