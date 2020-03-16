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
from delta.layers import utils_tf as utils
import delta.layers

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments, too-many-locals

class TransformerEncoderLayer(Layer):
  """
  TransformerEncoderLayer is based on "Attention
  is all you Need" (https://arxiv.org/pdf/1706.03762.pdf).
  """
  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    model_config = config['model']['net']['structure']
    self.head_num = model_config.get('head_num')
    self.hidden_dim = model_config.get('hidden_dim')
    self.inner_size = model_config.get('inner_size')
    self.feed_forward_act = config.get('feed_forward_act', 'relu')
    self.dropout_rate = config.get('dropout_rate', 0.)

    self.self_attn_layer = delta.layers.MultiHeadAttention(
      self.hidden_dim, self.head_num)
    self.feed_forward_layer = delta.layers.PositionwiseFeedForward(
      self.hidden_dim, self.inner_size, self.feed_forward_act)
    self.embed_dense = tf.keras.layers.Dense(self.hidden_dim)
    
    self.self_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.feed_forward_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    
    self.self_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.feed_forward_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inps, training=None, mask=None):
    '''
    Input shape: [batch_size, seq_enc_len, hidden_dim]
    Mask shape: [batch_size, seq_enc_len]
    '''
    # Multi Head Attention
    inps = self.embed_dense(inps)
    self_attn_outs, _ = self.self_attn_layer((inps, inps, inps),
                                          training=training,
                                             mask=mask)
    self_attn_outs = self.self_attn_dropout(
        self_attn_outs, training=training)
    self_attn_outs += inps
    self_attn_outs = self.self_attn_norm(self_attn_outs)
    # Position Wise Feed Forward
    feed_forward_outs = self.feed_forward_layer(self_attn_outs)
    feed_forward_outs = self.feed_forward_dropout(
        feed_forward_outs, training=training)
    feed_forward_outs += self_attn_outs
    outs = self.feed_forward_norm(feed_forward_outs)
    return outs


class TransformerDecoderLayer(Layer):
  """
  TransformerEncoderLayer is based on "Attention
  is all you Need" (https://arxiv.org/pdf/1706.03762.pdf).
  """
  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    model_config = config['model']['net']['structure']
    self.head_num = model_config.get('head_num')
    self.hidden_dim = model_config.get('hidden_dim')
    self.inner_size = model_config.get('inner_size')
    self.feed_forward_act = config.get('feed_forward_act', 'relu')
    self.dropout_rate = config.get('dropout_rate', 0.)

    self.self_attn_layer = delta.layers.MultiHeadAttention(
      self.hidden_dim, self.head_num)
    self.context_attn_layer = delta.layers.MultiHeadAttention(
      self.hidden_dim, self.head_num)
    self.feed_forward_layer = delta.layers.PositionwiseFeedForward(
      self.hidden_dim, self.inner_size, self.feed_forward_act)
    self.enc_embed_dense = tf.keras.layers.Dense(self.hidden_dim)
    self.dec_embed_dense = tf.keras.layers.Dense(self.hidden_dim)

    self.self_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.context_attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.feed_forward_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    self.self_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.context_attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.feed_forward_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inps, training=None, mask=None):
    '''
    Input: [dec_inps, enc_outs]
    Mask: [dec_mask, enc_mask]

    Input shape: [[batch_size, seq_dec_len, hidden_dim],
                    (batch_size, seq_enc_len, hidden_dim)]
    Mask shape: [(batch_size, seq_dec_len), (batch_size, seq_enc_len)]
    '''
    dec_inps, enc_outs = inps
    dec_inps = self.dec_embed_dense(dec_inps)
    enc_outs = self.enc_embed_dense(enc_outs)
    look_ahead_mask, enc_mask = mask
    # Self Attention
    self_attn_outs, _ = self.self_attn_layer(
      (dec_inps, dec_inps, dec_inps),
      training=training,
      mask=look_ahead_mask)
    self_attn_outs = self.self_attn_dropout(
        self_attn_outs, training=training)
    self_attn_outs += dec_inps
    self_attn_outs = self.self_attn_norm(self_attn_outs)

    # Context Attention
    context_attn_outs, _ = self.context_attn_layer(
      (self_attn_outs, enc_outs, enc_outs),
      training=training,
      mask=enc_mask)
    context_attn_outs = self.context_attn_dropout(
        context_attn_outs, training=training)
    context_attn_outs += self_attn_outs
    context_attn_outs = self.context_attn_norm(context_attn_outs)

    # Position Wise Feed Forward
    feed_forward_outs = self.feed_forward_layer(context_attn_outs)
    feed_forward_outs = self.feed_forward_dropout(
        feed_forward_outs, training=training)
    feed_forward_outs = context_attn_outs + feed_forward_outs
    outs = self.feed_forward_norm(feed_forward_outs)

    return outs


class TransformerEncoder(Layer):
  """
  Transformer Encoder is stacked with
  several TransformerencLayers.
  """

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    model_config = config['model']['net']['structure']
    self.num_layers = model_config.get('num_layers')
    self.transformer_encs = [
        TransformerEncoderLayer(config) for _ in range(self.num_layers)
    ]

  def call(self, inps, training=None, mask=None):
    """
    Input shape: [Batch size, enc_seq_len]
    Output shape: [Batch size, enc_seq_len, hidden_dim]
    """
    enc_inps = inps
    for enc_layer in self.transformer_encs:
      enc_inps = enc_layer(enc_inps, training=training, mask=mask)
    enc_outs = enc_inps
    return enc_outs


class TransformerDecoder(Layer):
  """
  Transformer Encoder is stacked with several
  TransformerdecLayers, consist of beamsearch for infrence.
  """

  def __init__(self, config, embed_layer, vocab_size, **kwargs):
    super().__init__(**kwargs)
    model_config = config['model']['net']['structure']
    self.is_infer = config['model']['is_infer']
    if self.is_infer:
      self.length_penalty = model_config.get('length_penalty')
    self.dropout_rate = model_config.get('dropout_rate')
    self.num_layers = model_config.get('num_layers')
    self.embedding_size = model_config.get('embedding_size')
    self.max_dec_len = model_config.get('max_dec_len')
    self.padding_id = model_config.get('padding_id', 0)
    self.sos_id = model_config.get('sos_id', 4)
    self.eos_id = model_config.get('eos_id', 5)
    self.beam_size = model_config.get('beam_size')
    self.use_const = model_config.get('use_const', True)
    self.share_embedding = model_config.get('share_embedding', True)

    self.vocab_size = vocab_size
    self.embed_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    embed_layer, pos_embed_layer = embed_layer
    if self.share_embedding:
      self.pos_embed_layer = pos_embed_layer
    else:
      self.pos_embed_layer = delta.layers.PositionEmbedding(
        self.max_dec_len, self.embedding_size, self.use_const, "dec_pos")
    self.embed_layer = embed_layer
    self.transformer_decs = [
        TransformerDecoderLayer(config) for _ in range(self.num_layers)
    ]
    self.final_dense = tf.keras.layers.Dense(vocab_size)

  def decode(self, dec_inps, enc_out, training=None, mask=None):
    """
    Decoder func
    """
    look_ahead_mask = utils.create_look_ahead_mask(dec_inps)
    mask = (look_ahead_mask, mask)
    dec_emb = self.embed_layer(dec_inps)
    dec_pos_emb = self.pos_embed_layer(dec_inps)
    dec_emb += dec_pos_emb

    dec_inp = dec_emb
    for dec_layer in self.transformer_decs:
      dec_inp = dec_layer([dec_inp, enc_out],
                          training=training,
                          mask=mask)
    dec_out = dec_inp
    return dec_out

  def call(self, inps, training=None, mask=None):
    if not self.is_infer:
      dec_inp, enc_out = inps
      with tf.name_scope('while'):
        dec_out = self.decode(dec_inp, enc_out, training, mask)
        scores = self.final_dense(dec_out)
        return scores
    else:
      enc_out = inps
      init_ids = tf.cast(tf.ones([utils.shape_list(enc_out)[0]]) * self.sos_id, tf.int32)
      # Beam Search
      enc_shape = utils.shape_list(enc_out)
      enc_out = tf.tile(
          tf.expand_dims(enc_out, axis=1), [1, self.beam_size, 1, 1])
      enc_out = tf.reshape(
          enc_out, [enc_shape[0] * self.beam_size, enc_shape[1], enc_shape[2]])
      enc_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.beam_size, 1, 1, 1])
      enc_mask = tf.reshape(enc_mask,
                            [enc_shape[0] * self.beam_size, 1, 1, -1])
      def symbols_to_logits_fn(dec_inps):
        dec_out = self.decode(dec_inps, enc_out, training, enc_mask)
        scores = self.final_dense(dec_out)
        return scores[:, -1, :]

      decoded_ids, scores, _ = self.beam_search(symbols_to_logits_fn, init_ids,
                                                self.beam_size,
                                                self.max_dec_len,
                                                self.vocab_size,
                                                self.length_penalty,
                                                self.eos_id)
      decoded_ids = decoded_ids[:, 0, 1:]

      return decoded_ids

  @staticmethod
  def beam_search(symbols_to_logits_fn,
                  initial_ids,
                  beam_size,
                  decode_length,
                  vocab_size,
                  alpha,
                  eos_id,
                  states=None,
                  stop_early=True,
                  INF=1. * 1e20):
    """Beam search with length penalties."""
    batch_size = utils.shape_list(initial_ids)[0]

    initial_log_probs = tf.constant([[0.] + [-INF] * (beam_size - 1)])
    # (batch_size, beam_size)
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])

    alive_seq = utils.expand_to_beam_size(initial_ids, beam_size)
    # (batch_size, beam_size, 1)
    alive_seq = tf.expand_dims(alive_seq, axis=2)
    if states:
      states = nest.map_structure(
          lambda state: utils.expand_to_beam_size(state, beam_size), states)
    else:
      states = {}

    # (batch_size, beam_size, 1)
    finished_seq = tf.zeros(utils.shape_list(alive_seq), tf.int32)
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
      return utils.compute_topk_scores_and_seq(curr_finished_seq,
                                               curr_finished_scores,
                                               curr_finished_scores,
                                               curr_finished_flags, beam_size,
                                               batch_size, "grow_finished")

    def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                   states):
      """Given sequences and scores, will gather the top k=beam size sequences."""
      curr_scores += tf.to_float(curr_finished) * -INF
      return utils.compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                               curr_finished, beam_size, batch_size,
                                               "grow_alive", states)

    def grow_topk(i, alive_seq, alive_log_probs, states):
      """Inner beam search loop."""
      flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])

      # (batch_size * beam_size, decoded_length)
      if states:
        flat_states = nest.map_structure(utils.merge_beam_dim, states)
        flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i,
                                                        flat_states)
        states = nest.map_structure(
            lambda t: utils.unmerge_beam_dim(t, batch_size, beam_size), flat_states)
      else:
        flat_logits = symbols_to_logits_fn(flat_ids)

      logits = tf.reshape(flat_logits, [batch_size, beam_size, -1])
      candidate_log_probs = utils.log_prob_from_logits(logits)
      log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

      length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)

      curr_scores = log_probs / length_penalty
      flat_curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])

      topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2)
      topk_log_probs = topk_scores * length_penalty

      topk_beam_index = topk_ids // vocab_size
      topk_ids %= vocab_size  # Unflatten the ids
      batch_pos = utils.compute_batch_indices(batch_size, beam_size * 2)
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

    state_struc = nest.map_structure(utils.get_state_shape_invariants, states)
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
