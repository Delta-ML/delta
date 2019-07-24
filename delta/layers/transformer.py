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
"""Transformer layers."""
import tensorflow as tf
import numpy as np
from delta.layers.base_layer import Layer

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments, too-many-locals


class MultiHeadAttention(Layer):
  """
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  def __init__(self,
               head_num,
               activation='relu',
               use_bias=False,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

    super().__init__(**kwargs)
    self.supports_masking = True
    self.head_num = head_num
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

  @staticmethod
  def _unpack(inputs):
    """unpack attention inputs, shape and mask"""
    query, key, value = inputs
    return query, key, value

  def build(self, input_shape):
    # pylint: disable=attribute-defined-outside-init
    query_shape, key_shape, value_shape = self._unpack(input_shape)

    feature_dim = int(value_shape[-1])
    if feature_dim % self.head_num != 0:
      error_info = 'Invalid head number {} with the given input dim {}'.format(
          self.head_num, feature_dim)
      logging.error(error_info)
      raise ValueError(error_info)

    self.kernel_query = self.add_weight(
        name='Wq',
        shape=(int(query_shape[-1]), feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.kernel_key = self.add_weight(
        name='Wk',
        shape=(int(key_shape[-1]), feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.kernel_value = self.add_weight(
        name='Wv',
        shape=(int(value_shape[-1]), feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    self.kernel_project = self.add_weight(
        name='Wo',
        shape=(feature_dim, feature_dim),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    if self.use_bias:
      self.b_query = self.add_weight(
          name='Bq',
          shape=(feature_dim,),
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

      self.b_key = self.add_weight(
          name='Bk',
          shape=(feature_dim,),
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

      self.b_value = self.add_weight(
          name='Bb',
          shape=(feature_dim,),
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

      self.b_project = self.add_weight(
          name='Bo',
          shape=(feature_dim,),
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

  def compute_mask(self, inputs, mask=None):
    query_mask, _, _ = self._unpack(mask)
    return query_mask

  def compute_output_shape(self, input_shape):
    query_shape, _, value_shape = self._unpack(input_shape)
    return query_shape[:-1] + (value_shape[-1],)

  def call(self, inputs, training=None, mask=None):

    query, key, value = self._unpack(inputs)

    query_mask, key_mask, _ = self._unpack(mask)

    batch_size = tf.shape(query)[0]
    dimension_query = query.get_shape().as_list()[-1]
    seq_len = tf.shape(query)[-2]
    feature_dim = tf.shape(value)[-1]

    query = tf.matmul(
        query,
        tf.tile(tf.expand_dims(self.kernel_query, 0), [batch_size, 1, 1]))
    key = tf.matmul(
        key, tf.tile(tf.expand_dims(self.kernel_key, 0), [batch_size, 1, 1]))
    value = tf.matmul(
        value,
        tf.tile(tf.expand_dims(self.kernel_value, 0), [batch_size, 1, 1]))
    if self.use_bias:
      query += self.b_query
      key += self.b_key
      value += self.b_value

    def _reshape_multihead(origin_input):
      """
      reshape for multi head
        Input shape: (Batch size, steps, features)
        Output shape: (Batch size * head num, steps, features // head num)
      """
      return tf.concat(tf.split(origin_input, self.head_num, axis=2), axis=0)

    def _reshape_mask(mask):
      """
      repeat mask for multi head
        Input shape: (Batch size, steps)
        Output shape: (Batch size * head num, steps)
      """
      if mask is None:
        return None
      seq_len = tf.shape(mask)[1]
      mask = tf.expand_dims(mask, axis=1)
      mask = tf.tile(mask, [1, self.head_num, 1])
      return tf.reshape(mask, shape=(-1, seq_len))

    query_ = _reshape_multihead(query)
    key_ = _reshape_multihead(key)
    value_ = _reshape_multihead(value)

    key_mask = _reshape_mask(key_mask)

    # (Batch size * head num, query steps, key steps)
    similaritys = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))
    # scale
    similaritys /= tf.sqrt(tf.cast(dimension_query, tf.float32))
    if key_mask is not None:
      similaritys -= (
          1.0 - tf.cast(tf.expand_dims(key_mask, axis=-2), tf.float32)) * 1e9

    attention_weights = tf.keras.activations.softmax(similaritys)
    attention_outputs = tf.matmul(attention_weights, value_)
    attention_outputs = tf.reshape(
        attention_outputs,
        (-1, self.head_num, seq_len, feature_dim // self.head_num))
    attention_outputs = tf.transpose(attention_outputs, [0, 2, 1, 3])
    attention_outputs = tf.reshape(attention_outputs,
                                   (-1, seq_len, feature_dim))

    attention_outputs = tf.matmul(
        attention_outputs,
        tf.tile(tf.expand_dims(self.kernel_project, 0), [batch_size, 1, 1]))
    if self.use_bias:
      attention_outputs += self.b_project
    if self.activation is not None:
      attention_outputs = self.activation(attention_outputs)

    attention_outputs *= (
        1.0 - tf.cast(tf.expand_dims(query_mask, axis=-1), tf.float32))

    return attention_outputs


class MultiHeadSelfAttention(MultiHeadAttention):
  """
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  @staticmethod
  def _unpack(inputs):
    """unpack attention inputs, shape and mask"""
    query = key = value = inputs
    return query, key, value


class LayerNormalization(Layer):
  """
  Refer to [Layer Normalization]
    (https://arxiv.org/pdf/1607.06450.pdf)
    Input shape: (Batch size, steps, features)
    Output shape: (Batch size, steps, features)
  """

  def __init__(self, eps=1e-8, **kwargs):
    self.eps = eps
    self.gamma, self.beta = None, None
    super().__init__(**kwargs)

  def build(self, input_shape):
    self.gamma = self.add_weight(
        name='gamma',
        shape=input_shape[-1:],
        initializer=tf.keras.initializers.ones,
        trainable=True)
    self.beta = self.add_weight(
        name='beta',
        shape=input_shape[-1:],
        initializer=tf.keras.initializers.zeros,
        trainable=True)
    super().build(input_shape)

  def call(self, inputs, training=None, mask=None):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / ((variance + self.eps)**.5)
    return self.gamma * normalized + self.beta

  def compute_output_shape(self, input_shape):
    return input_shape


class PositionEmbedding(Layer):
  """
  Position embedding use sine and cosine functions
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
    Input shape: (Batch size, steps, features)
    Output shape: (Batch size, steps, features)
  """

  def __init__(self, max_len, embedding_dim, **kwargs):
    self.max_len = max_len
    self.embedding_dim = embedding_dim
    self.pos_embedding_matrix = self.get_pos_embedding_matrix(
        self.max_len, self.embedding_dim)
    embed_initializer = tf.constant_initializer(self.pos_embedding_matrix)
    self.pos_embedding_layer = tf.keras.layers.Embedding(
        *self.pos_embedding_matrix.shape,
        trainable=False,
        embeddings_initializer=embed_initializer)
    self.get_pos_layer = tf.keras.layers.Lambda(self.get_pos)
    self.mask_layer = tf.keras.layers.Lambda(self.mask_outputs)
    super().__init__(**kwargs)

  @staticmethod
  def get_pos_embedding_matrix(max_len, embedding_dim):
    """get position embedding by sine and cosine functions"""
    # First part of the PE function: sin and cos argument
    position_enc = np.array([[
        pos / np.power(10000, (i - i % 2) / embedding_dim)
        for i in range(embedding_dim)
    ]
                             for pos in range(max_len)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc

  @staticmethod
  def get_pos(inputs):
    """get position id"""
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
    position_ind = tf.tile(
        tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
    return position_ind

  @staticmethod
  def mask_outputs(origin_outputs):
    """mask position embedding"""
    inputs, outputs = origin_outputs
    outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
    return outputs

  def call(self, inputs, training=None, mask=None):
    pos_ind = self.get_pos_layer(inputs)
    pos_embedding = self.pos_embedding_layer(pos_ind)
    pos_embedding = self.mask_layer([inputs, pos_embedding])
    return pos_embedding


class TransformerEncoder(Layer):
  """
  Transformer Encoder Layer
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  def __init__(self,
               head_num,
               hidden_dim,
               feature_dim,
               attention_activation=None,
               feed_forward_activation='relu',
               dropout_rate=0.0,
               residual_conn=False,
               **kwargs):
    self.head_num = head_num
    self.hidden_dim = hidden_dim
    self.feature_dim = feature_dim
    self.attention_activation = attention_activation
    self.feed_forward_activation = feed_forward_activation
    self.dropout_rate = dropout_rate
    self.residual_conn = residual_conn

    self.attention_layer = MultiHeadSelfAttention(self.head_num,
                                                  self.attention_activation)
    self.attention_dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
    self.hidden_layer = tf.keras.layers.Conv1D(
        self.hidden_dim, 1, activation=self.feed_forward_activation)
    self.feed_forward_layer = tf.keras.layers.Conv1D(feature_dim, 1)
    self.feed_forward_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    if self.residual_conn:
      self.attention_ln = LayerNormalization()
      self.feed_forward_ln = LayerNormalization()
    super().__init__(**kwargs)

  def call(self, inputs, training=None, mask=None):
    # Multi Head Attention
    attention_layer = self.attention_layer(inputs, training=training, mask=mask)
    attention_dropout_layer = self.attention_dropout_layer(
        attention_layer, training=training)
    if self.residual_conn:
      attention_out = tf.keras.layers.add([inputs, attention_dropout_layer])
      attention_out = self.attention_ln(attention_out)
    else:
      attention_out = attention_dropout_layer

    # Position Wise Feed Forward
    hidden_layer = self.hidden_layer(attention_out)
    feed_forward_layer = self.feed_forward_layer(hidden_layer)
    feed_forward_dropout = self.feed_forward_dropout(
        feed_forward_layer, training=training)
    if self.residual_conn:
      feed_forward_out = tf.keras.layers.add(
          [attention_out, feed_forward_dropout])
      feed_forward_out = self.feed_forward_ln(feed_forward_out)
    else:
      feed_forward_out = feed_forward_layer

    return feed_forward_out
