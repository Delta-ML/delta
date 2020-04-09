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
"""Transformer sub layers."""
from absl import logging
import delta.compat as tf
import numpy as np
from delta.layers.base_layer import Layer
from delta.layers.utils_tf import shape_list

#pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments


class PositionEmbedding(Layer):
  """
    PositionEmbedding represents the positional information of tokens
    consisting of two optional types: constant(untrainable) and trainable.
  """
  def __init__(self, max_len, embed_dim, use_const, name, **kwargs):
    super().__init__(**kwargs)
    self.max_len = max_len
    self.embed_dim = embed_dim
    self.use_const = use_const
    self.pos_name = name
    self.pos_embed = self.get_pos_embedding_matrix(self.max_len,
                                                   self.embed_dim,
                                                   self.use_const,
                                                   self.pos_name)
  @staticmethod
  def get_pos_embedding_matrix(max_len, embed_dim, use_const, name):
    """
    generate position embedding matrix, two optional types:
    constant(untrainable) and trainable.
    Args:
      max_len, embed_dim, use_const

    Return:
      pos_embed: [max_len, embed_dim]
    """
    # First part of the PE function: sin and cos argument
    if use_const:
      pos_embed = np.array([[
        pos / np.power(10000, (i - i % 2) / embed_dim)
        for i in range(embed_dim)
      ] for pos in range(max_len)])

      # Second part, apply the cosine to even columns and sin to odds.
      pos_embed[:, 0::2] = np.sin(pos_embed[:, 0::2])  # dim 2i
      pos_embed[:, 1::2] = np.cos(pos_embed[:, 1::2])  # dim 2i+1
      pos_embed = pos_embed[np.newaxis, ...]
      pos_embed = tf.cast(pos_embed, dtype=tf.float32)
    else:
      pos_embed = tf.get_variable(
          name=name,
          shape=[max_len, embed_dim],
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
      pos_embed = tf.expand_dims(pos_embed, 0)

    return pos_embed

  def call(self, inputs, training=None, mask=None):
    """
    Args:
       inputs: [batch_size, seq_x_len, embed_dim]
    Return:
      pos_embed: [batch_size, seq_x_len, embed_dim]
    """
    seq_len = shape_list(inputs)[1]
    pos_embed = self.pos_embed[:, :seq_len, :]
    return pos_embed


class PositionwiseFeedForward(Layer):
  """
  A two-layer Feed-Forward-Network.
  """
  def __init__(self, d_model, dff, act_func, **kwargs):
    super().__init__(**kwargs)
    self.dense1 = tf.keras.layers.Dense(dff, activation=act_func)
    self.dense2 = tf.keras.layers.Dense(d_model)

  def call(self, inputs, training=None, mask=None):
    """
    The implementation of PositionwiseFeedForward.
    Args:
      inputs: [batch_size, seq_x_len, d_model]
    Return:
      ffn: [batch_size, seq_x_len, d_model]
    """
    ffn = self.dense2(self.dense1(inputs))
    return ffn


class MultiHeadAttention(Layer):
  """
   Multi-headed attention is based on "Attention
  is all you Need" (https://arxiv.org/pdf/1706.03762.pdf).
  """
  def __init__(self, hidden_size, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.hidden_size, self.num_heads = hidden_size, num_heads
    assert self.hidden_size % self.num_heads == 0

    self.depth = self.hidden_size // self.num_heads

    self.wq = tf.keras.layers.Dense(self.hidden_size)
    self.wk = tf.keras.layers.Dense(self.hidden_size)
    self.wv = tf.keras.layers.Dense(self.hidden_size)

    self.dense = tf.keras.layers.Dense(self.hidden_size)

  def split_heads(self, x, batch_size):
    """
    Split hidden_size into depth(hidden_size // num_heads) for
    multi-head attention.
    Args:
      x: (batch_size, seq_len_x, hidden_size)
      batch_size

    Returns:
      split_x: (batch_size, num_heads, seq_len_x, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    split_x = tf.transpose(x, perm=[0, 2, 1, 3])
    return split_x

  def call(self, inputs, training=None, mask=None):
    """
    The implementation of Multi-headed attention.
    Args:
      inputs = (v, k, q)
      q: (batch_size, seq_len_q, hidden_size)
      k: (batch_size, seq_len_k, hidden_size)
      v: (batch_size, seq_len_v, hidden_size)
      mask: (batch_size, seq_len_q, seq_len_k)

    Returns:
      output: (batch_size, seq_len_q, hidden_size)
      attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    q, k, v = inputs
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len_q, hidden_size)
    k = self.wk(k)  # (batch_size, seq_len_k, hidden_size)
    v = self.wv(v)  # (batch_size, seq_len_v, hidden_size)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = self.scaled_dot_product_attention(
      q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_q, hidden_size)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, hidden_size)

    return output, attention_weights

  @staticmethod
  def scaled_dot_product_attention(q, k, v, mask):
    """
    The implementation of scaled attention.
    Args:
      v: (batch_size, seq_len_v, hidden_size)
      k: (batch_size, seq_len_k, hidden_size)
      q: (batch_size, seq_len_q, hidden_size)
      mask: (batch_size, seq_len_q, seq_len_k)

    Returns:
      output: (batch_size, seq_len_q, hidden_size)
      attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, seq_len_q, seq_len_k)

    # Scaled
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Masked
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)

    # Normalized
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, seq_len_q, seq_len_k)

    # Weighted sum
    output = tf.matmul(attention_weights, v)  # (batch_size, seq_len_q, depth_v)

    return output, attention_weights
