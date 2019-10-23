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
"""Recurrent neural network layers."""

from absl import logging
import delta.compat as tf

import delta
from delta.layers.base_layer import Layer
from tensorflow_addons import seq2seq
SOS_ID = 4
EOS_ID = 5


class BiRnn(Layer):
  """
  Bidirectional RNN
  Input Shape: [batch_size, steps, features]
  Output Shape: [batch_size, units]
  """

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    logging.info("Initialize Rnn {}...".format(self.name))

    model_config = config['model']['net']['structure']
    self.cell_dim = model_config['cell_dim']
    self.cell_type = model_config['cell_type']
    if self.cell_type.lower() == 'gru':
      rnn_class = tf.keras.layers.GRU
    elif self.cell_type.lower() == 'lstm':
      rnn_class = tf.keras.layers.LSTM
    elif self.cell_type.lower() == 'cudnngru':
      rnn_class = tf.keras.layers.CuDNNGRU
    elif self.cell_type.lower() == 'cudnnlstm':
      rnn_class = tf.keras.layers.CuDNNLSTM
    else:
      error_info = "Cell type: {} not supported now! Please check!".format(
          self.cell_type)
      logging.error(error_info)
      raise ValueError(error_info)

    self.bi_rnn = tf.keras.layers.Bidirectional(
        rnn_class(self.cell_dim, return_sequences=True))
    logging.info("Initialize Rnn {} Done.".format(self.name))

  def build(self, input_shape):
    self.built = True

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], self.cell_dim * 2])
  
  def compute_mask(self, inputs, mask=None):
    return None

  def call(self, inputs, training=None, mask=None):
    out = self.bi_rnn(inputs)
    return out


class RnnAttentionEncoder(Layer):  # pylint: disable=too-many-instance-attributes
  """
  RNN + Attention
  Input Shape: [batch_size, steps, features]
  Output Shape: [batch_size, units]
  """

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    logging.info("Initialize RnnAttentionEncoder {}...".format(self.name))

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.cell_dim = model_config['cell_dim']

    self.sen_encoder = BiRnn(config)
    # self.sen_all_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200))
    self.sen_att = delta.layers.HanAttention(name="{}_att".format(self.name))
    self.sen_att_d = tf.keras.layers.Dropout(self.dropout_rate)
    # self.sen_att_d_bn = tf.keras.layers.BatchNormalization()
    logging.info("Initialize RnnAttentionEncoder {} Done.".format(self.name))

  def build(self, input_shape):
    self.built = True

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], self.cell_dim * 2])

  def compute_mask(self, inputs, mask=None):
    return None

  def call(self, inputs, training=None, mask=None):
    out = self.sen_encoder(inputs)
    # out = self.sen_all_dense(out)
    out = self.sen_att(out, mask=mask)
    out = self.sen_att_d(out, training=training)
    # out = self.sen_att_d_bn(out, training=training)
    return out


class RnnEncoder(Layer):  # pylint: disable=too-many-instance-attributes
  """
  RNN + Attention
  Input Shape: [batch_size, steps, features]
  Output Shape: [batch_size, units]
  """

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    logging.info("Initialize RnnEncoder {}...".format(self.name))

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.cell_dim = model_config['cell_dim']
    self.cell_type = model_config['cell_type']
    if self.cell_type.lower() == 'gru':
      rnn_class = tf.keras.layers.GRU
    elif self.cell_type.lower() == 'lstm':
      rnn_class = tf.keras.layers.LSTM
    elif self.cell_type.lower() == 'cudnngru':
      rnn_class = tf.keras.layers.CuDNNGRU
    elif self.cell_type.lower() == 'cudnnlstm':
      rnn_class = tf.keras.layers.CuDNNLSTM
    else:
      error_info = "Cell type: {} not supported now! Please check!".format(
          self.cell_type)
      logging.error(error_info)
      raise ValueError(error_info)

    self.sen_encoder = tf.keras.layers.Bidirectional(
        rnn_class(self.cell_dim, return_sequences=True, return_state=True))
    logging.info("Initialize RnnEncoder {} Done.".format(self.name))

  def build(self, input_shape):
    self.built = True

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], self.cell_dim * 2])

  def call(self, inputs, training=None, mask=None):
    if 'lstm' in self.cell_type.lower():
      out, forward_h, forward_c, backward_h, backward_c = self.sen_encoder(
          inputs)
      state_h = tf.keras.layers.concatenate([forward_h, backward_h])
      state_c = tf.keras.layers.concatenate([forward_c, backward_c])
      states = tf.contrib.rnn.LSTMStateTuple(state_h, state_c)
    else:
      out, forward_h, backward_h = self.sen_encoder(inputs)
      states = tf.keras.layers.concatenate([forward_h, backward_h])
    return out, states


class RnnDecoder(Layer):  # pylint: disable=too-many-instance-attributes
  """
  RNN + Attention
  Input Shape: [batch_size, steps, features]
  Output Shape: [batch_size, units]
  """

  def __init__(self, config, emb_layer, vocab_size, **kwargs):
    super().__init__(**kwargs)
    logging.info("Initialize RnnDecoder {}...".format(self.name))
    self.is_infer = config['model']['is_infer']
    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.cell_dim = model_config['cell_dim']
    self.decode_cell_type = model_config['decode_cell_type']
    self.max_dec_len = model_config['max_dec_len']
    self.dec_end_id = 5
    self.dec_start_id = 4
    self.beam_size = model_config['beam_size']
    self.length_penalty = model_config['length_penalty']
    self.swap_memory = model_config['swap_memory']
    self.time_major = model_config['time_major']
    self.initial_decode_state = model_config['initial_decode_state']
    self.attn_Type = model_config['attn_Type']
    if self.decode_cell_type.lower() == 'gru':
      rnn_class = tf.nn.rnn_cell.GRUCell
    elif self.decode_cell_type.lower() == 'lstm':
      rnn_class = tf.nn.rnn_cell.LSTMCell
    else:
      error_info = "Cell type: {} not supported now! Please check!".format(
          self.decode_cell_type)
      logging.error(error_info)
      raise ValueError(error_info)

    self.cell = rnn_class(2 * self.cell_dim)
    self.embed = emb_layer
    self.vocab_size = vocab_size
    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    self.built = True

  def call(self, inputs, training=None, mask=None):
    dec_emb_fn = lambda ids: self.embed(ids)
    if self.is_infer:
      enc_outputs, enc_state, enc_seq_len = inputs
      batch_size = tf.shape(enc_outputs)[0]
      helper = seq2seq.GreedyEmbeddingHelper(
          embedding=dec_emb_fn,
          start_tokens=tf.fill([batch_size], self.dec_start_id),
          end_token=self.dec_end_id)
    else:
      dec_inputs, dec_seq_len, enc_outputs, enc_state, \
      enc_seq_len = inputs
      batch_size = tf.shape(enc_outputs)[0]
      dec_inputs = self.embed(dec_inputs)
      helper = seq2seq.TrainingHelper(
          inputs=dec_inputs, sequence_length=dec_seq_len)

    if self.is_infer and self.beam_size > 1:
      tiled_enc_outputs = seq2seq.tile_batch(
          enc_outputs, multiplier=self.beam_size)
      tiled_seq_len = seq2seq.tile_batch(enc_seq_len, multiplier=self.beam_size)
      attn_mech = self._build_attention(
          enc_outputs=tiled_enc_outputs, enc_seq_len=tiled_seq_len)
      dec_cell = seq2seq.AttentionWrapper(self.cell, attn_mech)
      tiled_enc_last_state = seq2seq.tile_batch(
          enc_state, multiplier=self.beam_size)
      tiled_dec_init_state = dec_cell.zero_state(
          batch_size=batch_size * self.beam_size, dtype=tf.float32)
      if self.initial_decode_state:
        tiled_dec_init_state = tiled_dec_init_state.clone(
            cell_state=tiled_enc_last_state)

      dec = seq2seq.BeamSearchDecoder(
          cell=dec_cell,
          embedding=dec_emb_fn,
          start_tokens=tf.tile([self.dec_start_id], [batch_size]),
          end_token=self.dec_end_id,
          initial_state=tiled_dec_init_state,
          beam_width=self.beam_size,
          output_layer=tf.layers.Dense(self.vocab_size),
          length_penalty_weight=self.length_penalty)
    else:
      attn_mech = self._build_attention(
          enc_outputs=enc_outputs, enc_seq_len=enc_seq_len)
      dec_cell = seq2seq.AttentionWrapper(
          cell=self.cell, attention_mechanism=attn_mech)
      dec_init_state = dec_cell.zero_state(
          batch_size=batch_size, dtype=tf.float32)
      if self.initial_decode_state:
        dec_init_state = dec_init_state.clone(cell_state=enc_state)
      dec = seq2seq.BasicDecoder(
          cell=dec_cell,
          helper=helper,
          initial_state=dec_init_state,
          output_layer=tf.layers.Dense(self.vocab_size))
    if self.is_infer:
      dec_outputs, _, _ = \
        seq2seq.dynamic_decode(decoder=dec,
                               maximum_iterations=self.max_dec_len,
                               swap_memory=self.swap_memory,
                               output_time_major=self.time_major)
      return dec_outputs.predicted_ids[:, :, 0]
    else:
      dec_outputs, _, _ = \
        seq2seq.dynamic_decode(decoder=dec,
                               maximum_iterations=tf.reduce_max(dec_seq_len),
                               swap_memory=self.swap_memory,
                               output_time_major=self.time_major)
    return dec_outputs.rnn_output

  def _build_attention(self, enc_outputs, enc_seq_len):
    with tf.variable_scope("AttentionMechanism"):
      if self.attn_Type == 'bahdanau':
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=2 * self.cell_dim,
            memory=enc_outputs,
            memory_sequence_length=enc_seq_len,
            probability_fn=tf.nn.softmax,
            normalize=True,
            dtype=tf.get_variable_scope().dtype)
      elif self.params['attention_type'] == 'luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=2 * self.cell_dim,
            memory=enc_outputs,
            memory_sequence_length=enc_seq_len,
            probability_fn=tf.nn.softmax,
            dtype=tf.get_variable_scope().dtype)
      else:
        raise ValueError('Unknown Attention Type')
      return attention_mechanism
