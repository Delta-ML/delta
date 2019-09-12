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
import math

from absl import logging
import numpy as np
import delta.compat as tf
from tensorflow.python.util import nest

from delta.layers.base_layer import Layer

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments, too-many-locals

SOS_ID = 4
EOS_ID = 5
INF = 1. * 1e7


def log_prob_from_logits(logits, reduce_axis=-1):
  """return log prob use log sum func"""
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)


def shape_list(tensor):
  """Return list of dims, statically where possible."""
  tensor = tf.convert_to_tensor(tensor)

  if tensor.get_shape().dims is None:
    return tf.shape(tensor)

  static = tensor.get_shape().as_list()
  shape = tf.shape(tensor)

  ret = []
  for i, _ in enumerate(static):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def _merge_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension."""
  shape = shape_list(tensor)
  shape[0] *= shape[1]  # batch -> batch * beam_size
  shape.pop(1)  # Remove beam dim
  return tf.reshape(tensor, shape)


def _unmerge_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size]."""
  shape = shape_list(tensor)
  new_shape = [batch_size] + [beam_size] + shape[1:]
  return tf.reshape(tensor, new_shape)


def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size."""
  tensor = tf.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf.tile(tensor, tile_dims)


def get_state_shape_invariants(tensor):
  """Returns the shape of the tensor but sets middle dims to None."""
  shape = tensor.shape.as_list()
  for i in range(1, len(shape) - 1):
    shape[i] = None
  return tf.TensorShape(shape)


def compute_batch_indices(batch_size, beam_size):
  """Computes the i'th coordinate that contains the batch index for gathers."""
  batch_pos = tf.range(batch_size * beam_size) // beam_size
  batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
  return batch_pos


def _create_make_unique(inputs):
  """Replaces the lower bits of each element with iota."""
  if inputs.shape.ndims != 2:
    raise ValueError("Input of top_k_with_unique must be rank-2 "
                     "but got: %s" % inputs.shape)

  height = inputs.shape[0]
  width = inputs.shape[1]
  zeros = tf.zeros([height, width], dtype=tf.int32)

  log2_ceiling = int(math.ceil(math.log(int(width), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = ~(next_power_of_two - 1)
  count_mask_r0 = tf.constant(count_mask)
  count_mask_r2 = tf.fill([height, width], count_mask_r0)

  smallest_normal = 1 << 23
  smallest_normal_r0 = tf.constant(smallest_normal, dtype=tf.int32)
  smallest_normal_r2 = tf.fill([height, width], smallest_normal_r0)

  low_bit_mask = ~(1 << 31)
  low_bit_mask_r0 = tf.constant(low_bit_mask, dtype=tf.int32)
  low_bit_mask_r2 = tf.fill([height, width], low_bit_mask_r0)

  iota = tf.tile(
      tf.expand_dims(tf.range(width, dtype=tf.int32), 0), [height, 1])

  input_r2 = tf.bitcast(inputs, tf.int32)
  abs_r2 = tf.bitwise.bitwise_and(input_r2, low_bit_mask_r2)
  if_zero_r2 = tf.equal(abs_r2, zeros)
  smallest_normal_preserving_sign_r2 = tf.bitwise.bitwise_or(
      input_r2, smallest_normal_r2)
  input_no_zeros_r2 = tf.where(if_zero_r2, smallest_normal_preserving_sign_r2,
                               input_r2)

  and_r2 = tf.bitwise.bitwise_and(input_no_zeros_r2, count_mask_r2)
  or_r2 = tf.bitwise.bitwise_or(and_r2, iota)
  return tf.bitcast(or_r2, tf.float32)


def _create_topk_unique(inputs, k):
  """Creates the top k values in sorted order with indices."""
  height = inputs.shape[0]
  width = inputs.shape[1]
  neg_inf_r0 = tf.constant(-np.inf, dtype=tf.float32)
  ones = tf.ones([height, width], dtype=tf.float32)
  neg_inf_r2 = ones * neg_inf_r0
  inputs = tf.where(tf.is_nan(inputs), neg_inf_r2, inputs)

  tmp = inputs
  topk_r2 = tf.zeros([height, k], dtype=tf.float32)
  for i in range(k):
    kth_order_statistic = tf.reduce_max(tmp, axis=1, keepdims=True)
    k_mask = tf.tile(
        tf.expand_dims(tf.equal(tf.range(k), tf.fill([k], i)), 0), [height, 1])
    topk_r2 = tf.where(k_mask, tf.tile(kth_order_statistic, [1, k]), topk_r2)
    ge_r2 = tf.greater_equal(inputs, tf.tile(kth_order_statistic, [1, width]))
    tmp = tf.where(ge_r2, neg_inf_r2, inputs)

  log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = next_power_of_two - 1
  mask_r0 = tf.constant(count_mask)
  mask_r2 = tf.fill([height, k], mask_r0)
  topk_r2_s32 = tf.bitcast(topk_r2, tf.int32)
  topk_indices_r2 = tf.bitwise.bitwise_and(topk_r2_s32, mask_r2)
  return topk_r2, topk_indices_r2


def top_k_with_unique(inputs, k):
  """Finds the values and indices of the k largests entries."""
  unique_inputs = _create_make_unique(tf.cast(inputs, tf.float32))
  top_values, indices = _create_topk_unique(unique_inputs, k)
  top_values = tf.cast(top_values, inputs.dtype)
  return top_values, indices


def compute_topk_scores_and_seq(sequences,
                                scores,
                                scores_to_gather,
                                flags,
                                beam_size,
                                batch_size,
                                prefix="default",
                                states_to_gather=None):
  """Given sequences and scores, will gather the top k=beam size sequences."""
  _, topk_indexes = tf.nn.top_k(scores, k=beam_size)
  batch_pos = compute_batch_indices(batch_size, beam_size)
  top_coordinates = tf.stack([batch_pos, topk_indexes], axis=2)

  def gather(tensor, name):
    return tf.gather_nd(tensor, top_coordinates, name=(prefix + name))

  topk_seq = gather(sequences, "_topk_seq")
  topk_flags = gather(flags, "_topk_flags")
  topk_gathered_scores = gather(scores_to_gather, "_topk_scores")
  if states_to_gather:
    topk_gathered_states = nest.map_structure(
        lambda state: gather(state, "_topk_states"), states_to_gather)
  else:
    topk_gathered_states = states_to_gather

  return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


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
               sequence_mask=False,
               **kwargs):

    super().__init__(**kwargs)
    self.supports_masking = True
    self.head_num = head_num
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.sequence_mask = sequence_mask
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
    key_len = tf.shape(key)[-2]
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
    if self.sequence_mask:
      ones = tf.ones((seq_len, key_len))
      similaritys -= (ones - tf.matrix_band_part(ones, -1, 0)) * 1e9
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

    if query_mask is not None:
      attention_outputs *= tf.cast(
          tf.expand_dims(query_mask, axis=-1), tf.float32)

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


class TransformerEncoderLayer(Layer):
  """
  Transformer Encoder Block Layer
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  def __init__(self, config, **kwargs):
    model_config = config['model']['net']['structure']
    self.head_num = model_config.get('head_num')
    self.hidden_dim = model_config.get('hidden_dim')
    self.feature_dim = model_config.get('embedding_size')
    self.attention_activation = config.get('attention_activation', None)
    self.feed_forward_activation = config.get('feed_forward_activation', 'relu')
    self.dropout_rate = config.get('transformer_dropout', 0.)
    self.residual_conn = config.get('residual_conn', False)

    self.attention_layer = MultiHeadSelfAttention(self.head_num,
                                                  self.attention_activation)
    self.attention_dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
    self.hidden_layer = tf.keras.layers.Conv1D(
        self.hidden_dim, 1, activation=self.feed_forward_activation)
    self.feed_forward_layer = tf.keras.layers.Conv1D(self.feature_dim, 1)
    self.feed_forward_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    if self.residual_conn:
      self.attention_layernorm = LayerNormalization()
      self.feed_forward_layernorm = LayerNormalization()
    super().__init__(**kwargs)

  def call(self, inputs, training=None, mask=None):
    '''
    Input shape: (batch_size, enc_len, feature_dim)
    Mask shape: (batch_size, enc_len)
    '''
    # Multi Head Attention
    attention_layer = self.attention_layer(inputs, training=training, mask=mask)
    attention_dropout_layer = self.attention_dropout_layer(
        attention_layer, training=training)
    if self.residual_conn:
      attention_out = tf.keras.layers.add([inputs, attention_dropout_layer])
      attention_out = self.attention_layernorm(attention_out)
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
      feed_forward_out = self.feed_forward_layernorm(feed_forward_out)
    else:
      feed_forward_out = feed_forward_layer

    return feed_forward_out


class TransformerDecoderLayer(Layer):
  """
  Transformer Decoder Block Layer
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  def __init__(self, config, **kwargs):
    model_config = config['model']['net']['structure']
    self.head_num = model_config.get('head_num')
    self.hidden_dim = model_config.get('hidden_dim')
    self.feature_dim = model_config.get('embedding_size')
    self.attention_activation = config.get('attention_activation', None)
    self.feed_forward_activation = config.get('feed_forward_activation', 'relu')
    self.dropout_rate = config.get('transformer_dropout', 0.)

    # Self Attention
    self.self_attention_layer = MultiHeadSelfAttention(
        self.head_num, self.attention_activation, sequence_mask=True)
    self.self_attention_dropout_layer = tf.keras.layers.Dropout(
        self.dropout_rate)
    self.self_attention_ln = LayerNormalization()

    # Context Attention
    self.context_attention_layer = MultiHeadAttention(self.head_num,
                                                      self.attention_activation)
    self.context_attention_dropout_layer = tf.keras.layers.Dropout(
        self.dropout_rate)
    self.context_attention_ln = LayerNormalization()

    # Position Wise Feed Forward
    self.hidden_layer = tf.keras.layers.Conv1D(
        self.hidden_dim, 1, activation=self.feed_forward_activation)
    self.feed_forward_layer = tf.keras.layers.Conv1D(self.feature_dim, 1)
    self.feed_forward_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.feed_forward_ln = LayerNormalization()

    super().__init__(**kwargs)

  def call(self, inputs, training=None, mask=None):
    '''
    Input: [decoder_inputs, encoder_outputs]
    Mask: [decoder_mask, encoder_mask]

    Input shape: [(batch_size, dec_len, feature_dim), (batch_size, enc_len, feature_dim)]
    Mask shape: [(batch_size, dec_len), (batch_size, enc_len)]
    '''
    decoder_inputs, encoder_outputs = inputs
    decoder_mask, encoder_mask = mask
    # Self
    self_attention = self.self_attention_layer(
        decoder_inputs, training=training, mask=decoder_mask)
    self_attention_dropout = self.self_attention_dropout_layer(
        self_attention, training=training)
    self_attention_out = tf.keras.layers.add(
        [decoder_inputs, self_attention_dropout])
    self_attention_out = self.self_attention_ln(self_attention_out)

    # Context Attention
    context_attention = self.context_attention_layer(
        [self_attention_out, encoder_outputs, encoder_outputs],
        mask=[decoder_mask, encoder_mask, encoder_mask],
        training=training)
    context_attention_dropout = self.context_attention_dropout_layer(
        context_attention, training=training)
    context_attention_out = tf.keras.layers.add(
        [self_attention_out, context_attention_dropout])
    context_attention_out = self.self_attention_ln(context_attention_out)

    # Position Wise Feed Forward
    hidden_layer = self.hidden_layer(context_attention_out)
    feed_forward_layer = self.feed_forward_layer(hidden_layer)
    feed_forward_dropout = self.feed_forward_dropout(
        feed_forward_layer, training=training)
    feed_forward_out = tf.keras.layers.add(
        [context_attention_out, feed_forward_dropout])
    feed_forward_out = self.feed_forward_ln(feed_forward_out)

    return feed_forward_out


class TransformerEncoder(Layer):
  """
  Transformer Encoder Layer
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, steps, features)
  """

  def __init__(self, config, **kwargs):
    model_config = config['model']['net']['structure']
    self.is_infer = config['model']['is_infer']
    if self.is_infer:
      self.length_penalty = model_config['length_penalty']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.max_enc_len = model_config['max_enc_len']
    self.max_dec_len = model_config['max_dec_len']
    self.share_embedding = model_config['share_embedding']
    self.padding_token = 0
    self.beam_size = model_config['beam_size']

    self.transformer_encoders = [
        TransformerEncoderLayer(config) for _ in range(self.num_layers)
    ]

    super().__init__(**kwargs)

  def call(self, inputs, training=None, mask=None):
    enc_inp = inputs
    for encoder_layer in self.transformer_encoders:
      enc_inp = encoder_layer(enc_inp, training=training, mask=mask)
    enc_out = enc_inp
    return enc_out


class TransformerDecoder(Layer):
  """
  Transformer Decoder Layer
  Refer to [Attention Is All You Need]
    (https://arxiv.org/abs/1706.03762)
  Input shape: (Batch size, steps, features)
  Output shape: (Batch size, vocab_size)
  """

  def __init__(self, config, emb_layer, vocab_size, **kwargs):
    model_config = config['model']['net']['structure']
    self.is_infer = config['model']['is_infer']
    if self.is_infer:
      self.length_penalty = model_config['length_penalty']
    self.dropout_rate = model_config['dropout_rate']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.embedding_size = model_config['embedding_size']
    self.max_enc_len = model_config['max_enc_len']
    self.max_dec_len = model_config['max_dec_len']
    self.share_embedding = model_config['share_embedding']
    self.padding_token = 0
    self.beam_size = model_config['beam_size']

    self.mask_layer = tf.keras.layers.Lambda(lambda inputs: tf.cast(
        tf.not_equal(inputs, self.padding_token), tf.int32))

    self.embed = emb_layer
    self.vocab_size = vocab_size
    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)

    self.pos_embed = PositionEmbedding(self.max_enc_len, self.embedding_size)

    self.transformer_decoders = [
        TransformerDecoderLayer(config) for _ in range(self.num_layers)
    ]

    self.final_dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(self.vocab_size, name="final_dense"))

    super().__init__(**kwargs)

  def decode(self, input_dec_x, enc_out, enc_mask, training=None):
    """
    Decoder func
    """
    dec_mask = self.mask_layer(input_dec_x)
    dec_emb = self.embed(input_dec_x)
    dec_pos_emb = self.pos_embed(dec_emb)
    dec_emb = tf.keras.layers.add([dec_emb, dec_pos_emb])

    dec_inp = dec_emb
    for decoder_layer in self.transformer_decoders:
      dec_inp = decoder_layer([dec_inp, enc_out],
                              training=training,
                              mask=[dec_mask, enc_mask])
    dec_out = dec_inp
    return dec_out

  def call(self, inputs, training=None, mask=None):
    if not self.is_infer:
      dec_inp, enc_out = inputs
      with tf.name_scope('while'):
        dec_out = self.decode(dec_inp, enc_out, mask, training)
        scores = self.final_dense(dec_out)
        return scores
    else:
      enc_out = inputs
      init_ids = tf.cast(tf.ones([tf.shape(enc_out)[0]]) * SOS_ID, tf.int32)
      # Beam Search
      enc_shape = shape_list(enc_out)
      enc_out = tf.tile(
          tf.expand_dims(enc_out, axis=1), [1, self.beam_size, 1, 1])
      enc_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.beam_size, 1])
      enc_out = tf.reshape(
          enc_out, [enc_shape[0] * self.beam_size, enc_shape[1], enc_shape[2]])
      enc_mask = tf.reshape(enc_mask,
                            [enc_shape[0] * self.beam_size, enc_shape[1]])

      def symbols_to_logits_fn(dec_inputs):
        dec_out = self.decode(dec_inputs, enc_out, enc_mask, training)
        scores = self.final_dense(dec_out)
        return scores[:, -1, :]

      decoded_ids, scores, _ = self.beam_search(symbols_to_logits_fn, init_ids,
                                                self.beam_size,
                                                self.max_dec_len,
                                                self.vocab_size,
                                                self.length_penalty)
      decoded_ids = decoded_ids[:, 0, 1:]

      return decoded_ids

  @staticmethod
  def beam_search(symbols_to_logits_fn,
                  initial_ids,
                  beam_size,
                  decode_length,
                  vocab_size,
                  alpha,
                  states=None,
                  eos_id=EOS_ID,
                  stop_early=True):
    """Beam search with length penalties."""
    batch_size = shape_list(initial_ids)[0]

    initial_log_probs = tf.constant([[0.] + [-INF] * (beam_size - 1)])
    # Expand to beam_size (batch_size, beam_size)
    # (batch_size, beam_size)
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

    # Expand each batch and state to beam_size
    alive_seq = _expand_to_beam_size(initial_ids, beam_size)
    # (batch_size, beam_size, 1)
    alive_seq = tf.expand_dims(alive_seq, axis=2)
    if states:
      states = nest.map_structure(
          lambda state: _expand_to_beam_size(state, beam_size), states)
    else:
      states = {}

    # (batch_size, beam_size, 1)
    finished_seq = tf.zeros(shape_list(alive_seq), tf.int32)
    # Setting the scores of the initial to negative infinity.
    # (batch_size, beam_size)
    finished_scores = tf.ones([batch_size, beam_size]) * -INF
    # (batch_size, beam_size)
    finished_flags = tf.zeros([batch_size, beam_size], tf.bool)

    def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                      curr_scores, curr_finished):
      """
        Given sequences and scores from finished sequence and current finished sequence
        , will gather the top k=beam size sequences to update finished seq.
      """
      # padding zero for finished seq
      finished_seq = tf.concat(
          [finished_seq,
           tf.zeros([batch_size, beam_size, 1], tf.int32)],
          axis=2)

      # mask unfinished curr seq
      curr_scores += (1. - tf.to_float(curr_finished)) * -INF

      # concatenating the sequences and scores along beam axis
      # (batch_size, 2xbeam_size, seq_len)
      curr_finished_seq = tf.concat([finished_seq, curr_seq], axis=1)
      curr_finished_scores = tf.concat([finished_scores, curr_scores], axis=1)
      curr_finished_flags = tf.concat([finished_flags, curr_finished], axis=1)
      return compute_topk_scores_and_seq(curr_finished_seq,
                                         curr_finished_scores,
                                         curr_finished_scores,
                                         curr_finished_flags, beam_size,
                                         batch_size, "grow_finished")

    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                   states):
      """Given sequences and scores, will gather the top k=beam size sequences."""
      curr_scores += tf.to_float(curr_finished) * -INF
      return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                         curr_finished, beam_size, batch_size,
                                         "grow_alive", states)

    def grow_topk(i, alive_seq, alive_log_probs, states):
      """Inner beam search loop."""

      flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])

      # (batch_size * beam_size, decoded_length)
      if states:
        flat_states = nest.map_structure(_merge_beam_dim, states)
        flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i,
                                                        flat_states)
        states = nest.map_structure(
            lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_states)
      else:
        flat_logits = symbols_to_logits_fn(flat_ids)

      logits = tf.reshape(flat_logits, [batch_size, beam_size, -1])

      candidate_log_probs = log_prob_from_logits(logits)

      log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

      length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)

      curr_scores = log_probs / length_penalty
      flat_curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])

      topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2)

      topk_log_probs = topk_scores * length_penalty

      topk_beam_index = topk_ids // vocab_size
      topk_ids %= vocab_size  # Unflatten the ids
      batch_pos = compute_batch_indices(batch_size, beam_size * 2)
      topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2)

      topk_seq = tf.gather_nd(alive_seq, topk_coordinates)
      if states:
        states = nest.map_structure(
            lambda state: tf.gather_nd(state, topk_coordinates), states)
      topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

      topk_finished = tf.equal(topk_ids, eos_id)

      return topk_seq, topk_log_probs, topk_scores, topk_finished, states

    def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                   finished_flags, states):
      """Inner beam search loop."""
      topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
          i, alive_seq, alive_log_probs, states)
      alive_seq, alive_log_probs, _, states = grow_alive(
          topk_seq, topk_scores, topk_log_probs, topk_finished, states)
      finished_seq, finished_scores, finished_flags, _ = grow_finished(
          finished_seq, finished_scores, finished_flags, topk_seq, topk_scores,
          topk_finished)

      return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
              finished_flags, states)

    def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq,
                     finished_scores, unused_finished_in_finished,
                     unused_states):
      """Checking termination condition.
      """
      max_length_penalty = tf.pow(((5. + tf.to_float(decode_length)) / 6.),
                                  alpha)
      lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty

      if not stop_early:
        lowest_score_of_finished_in_finished = tf.reduce_min(finished_scores)
      else:
        lowest_score_of_finished_in_finished = tf.reduce_max(
            finished_scores, axis=1)

      bound_is_met = tf.reduce_all(
          tf.greater(lowest_score_of_finished_in_finished,
                     lower_bound_alive_scores))

      return tf.logical_and(
          tf.less(i, decode_length), tf.logical_not(bound_is_met))

    inner_shape = tf.TensorShape([None, None, None])

    state_struc = nest.map_structure(get_state_shape_invariants, states)
    (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
     finished_flags, states) = tf.while_loop(
         _is_finished,
         inner_loop, [
             tf.constant(0), alive_seq, alive_log_probs, finished_seq,
             finished_scores, finished_flags, states
         ],
         shape_invariants=[
             tf.TensorShape([]), inner_shape,
             alive_log_probs.get_shape(), inner_shape,
             finished_scores.get_shape(),
             finished_flags.get_shape(), state_struc
         ],
         parallel_iterations=1,
         back_prop=False)

    alive_seq.set_shape((None, beam_size, None))
    finished_seq.set_shape((None, beam_size, None))
    finished_seq = tf.where(
        tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
    finished_scores = tf.where(
        tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
    return finished_seq, finished_scores, states
