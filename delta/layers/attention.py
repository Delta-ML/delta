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
"""Attention layers."""
from absl import logging
import delta.compat as tf

from delta.layers.base_layer import Layer

#pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments


def masked_softmax(logits, mask, axis):
  """Compute softmax with input mask."""
  e_logits = tf.exp(logits)
  masked_e = tf.multiply(e_logits, mask)
  sum_masked_e = tf.reduce_sum(masked_e, axis, keep_dims=True)
  ones = tf.ones_like(sum_masked_e)
  # pay attention to a situation that if len of mask is zero,
  # denominator should be set to 1
  sum_masked_e_safe = tf.where(tf.equal(sum_masked_e, 0), ones, sum_masked_e)
  return masked_e / sum_masked_e_safe


class HanAttention(Layer):
  """
  Refer to [Hierarchical Attention Networks for Document Classification]
    (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    wrap `with tf.variable_scope(name, reuse=tf.AUTO_REUSE):`
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, features)
  """

  def __init__(self,
               W_regularizer=None,
               u_regularizer=None,
               b_regularizer=None,
               W_constraint=None,
               u_constraint=None,
               b_constraint=None,
               use_bias=True,
               **kwargs):

    super().__init__(**kwargs)
    self.supports_masking = True
    self.init = tf.keras.initializers.get('glorot_uniform')

    self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
    self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
    self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

    self.W_constraint = tf.keras.constraints.get(W_constraint)
    self.u_constraint = tf.keras.constraints.get(u_constraint)
    self.b_constraint = tf.keras.constraints.get(b_constraint)

    self.use_bias = use_bias

  def build(self, input_shape):
    # pylint: disable=attribute-defined-outside-init
    assert len(input_shape) == 3

    self.W = self.add_weight(
        name='{}_W'.format(self.name),
        shape=(
            int(input_shape[-1]),
            int(input_shape[-1]),
        ),
        initializer=self.init,
        regularizer=self.W_regularizer,
        constraint=self.W_constraint)

    if self.use_bias:
      self.b = self.add_weight(
          name='{}_b'.format(self.name),
          shape=(int(input_shape[-1]),),
          initializer='zero',
          regularizer=self.b_regularizer,
          constraint=self.b_constraint)

    self.attention_context_vector = self.add_weight(
        name='{}_att_context_v'.format(self.name),
        shape=(int(input_shape[-1]),),
        initializer=self.init,
        regularizer=self.u_regularizer,
        constraint=self.u_constraint)
    self.built = True

  # pylint: disable=missing-docstring, no-self-use
  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
    # do not pass the mask to the next layers
    return None

  def call(self, inputs, training=None, mask=None):
    batch_size = tf.shape(inputs)[0]
    W_3d = tf.tile(tf.expand_dims(self.W, axis=0), tf.stack([batch_size, 1, 1]))
    # [batch_size, steps, features]
    input_projection = tf.matmul(inputs, W_3d)

    if self.use_bias:
      input_projection += self.b

    input_projection = tf.tanh(input_projection)

    # [batch_size, steps, 1]
    similaritys = tf.reduce_sum(
        tf.multiply(input_projection, self.attention_context_vector),
        axis=2,
        keep_dims=True)

    # [batch_size, steps, 1]
    if mask is not None:
      attention_weights = masked_softmax(similaritys, mask, axis=1)
    else:
      attention_weights = tf.nn.softmax(similaritys, axis=1)

    # [batch_size, features]
    attention_output = tf.reduce_sum(
        tf.multiply(inputs, attention_weights), axis=1)
    return attention_output

  # pylint: disable=no-self-use
  def compute_output_shape(self, input_shape):
    """compute output shape"""
    return input_shape[0], input_shape[-1]


class MatchAttention(Layer):
  """
  Refer to [Learning Natural Language Inference with LSTM]
    (https://www.aclweb.org/anthology/N16-1170)
    wrap `with tf.variable_scope(name, reuse=tf.AUTO_REUSE):`
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    logging.info("Initialize MatchAttention {}...".format(self.name))
    self.fc_num_units = config['model']['net']['structure']['fc_num_units']
    self.middle_layer = tf.keras.layers.Dense(
        self.fc_num_units, activation='tanh')
    self.attn = tf.keras.layers.Dense(1)

  # pylint: disable=arguments-differ
  def call(self, tensors):
    """Attention layer."""
    left, right = tensors

    len_left = left.shape[1]
    len_right = right.shape[1]
    tensor_left = tf.expand_dims(left, axis=2)
    tensor_right = tf.expand_dims(right, axis=1)
    tensor_left = tf.tile(tensor_left, [1, 1, len_right, 1])
    tensor_right = tf.tile(tensor_right, [1, len_left, 1, 1])
    tensor_merged = tf.concat([tensor_left, tensor_right], axis=-1)
    middle_output = self.middle_layer(tensor_merged)
    attn_scores = self.attn(middle_output)
    attn_scores = tf.squeeze(attn_scores, axis=3)
    exp_attn_scores = tf.exp(attn_scores -
                             tf.reduce_max(attn_scores, axis=-1, keepdims=True))
    exp_sum = tf.reduce_sum(exp_attn_scores, axis=-1, keepdims=True)
    attention_weights = exp_attn_scores / exp_sum
    return tf.matmul(attention_weights, right)
