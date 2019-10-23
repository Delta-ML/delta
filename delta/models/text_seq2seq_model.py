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
"""Models for text sequence to sequence."""

import pickle

import delta.compat as tf
from absl import logging

from delta import layers
from delta import utils
from delta.models.base_model import Model
from delta.utils import registers
from delta.layers.utils import compute_sen_lens


class Seq2SeqModel(Model):
  """Base class for text sequence to sequence models"""

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    logging.info("Initialize S2SModel")
    data_config = config['data']
    model_config = config['model']['net']['structure']
    self.use_label_vocab = data_config['task']['use_label_vocab']
    self.label_vocab_size = data_config['label_vocab_size']
    self.vocab_size = config['data']['vocab_size']
    self.use_pretrained_embedding = config['model']['use_pre_train_emb']
    self.embedding_size = model_config['embedding_size']
    if self.use_pretrained_embedding:
      self.embedding_path = config['model']['embedding_path']
      logging.info("Loading embedding file from: {}".format(
          self.embedding_path))
      self._word_embedding_init = pickle.load(open(self.embedding_path, 'rb'))
      self.embed_initializer = tf.constant_initializer(
          self._word_embedding_init)
    else:
      self.embed_initializer = tf.random_uniform_initializer(-0.1, 0.1)

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        embeddings_initializer=self.embed_initializer)
    self.share_embedding = model_config['share_embedding']
    if self.use_label_vocab:
      self.decode_vocab_size = self.label_vocab_size
    else:
      self.decode_vocab_size = self.vocab_size
    if self.share_embedding:
      self.decoder_embed = self.embed
    else:
      self.decoder_embed = tf.keras.layers.Embedding(
          self.decode_vocab_size,
          self.embedding_size,
          embeddings_initializer=self.embed_initializer)


@registers.model.register
class TransformerSeq2SeqModel(Seq2SeqModel):
  """Transformer model for text sequence to sequence"""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    tf.logging.info("Initialize TransformerModel...")
    model_config = config['model']['net']['structure']
    self.is_infer = config['model']['is_infer']
    if self.is_infer:
      self.length_penalty = model_config['length_penalty']
    self.dropout_rate = model_config['dropout_rate']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.max_enc_len = model_config['max_enc_len']
    self.max_dec_len = model_config['max_dec_len']
    self.share_embedding = model_config['share_embedding']
    self.padding_token = utils.PAD_IDX
    self.beam_size = model_config['beam_size']

    self.mask_layer = tf.keras.layers.Lambda(lambda inputs: tf.cast(
        tf.not_equal(inputs, self.padding_token), tf.int32))

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)

    self.pos_embed = layers.PositionEmbedding(self.max_enc_len,
                                              self.embedding_size)

    self.encoder = layers.TransformerEncoder(config)
    self.decoder = layers.TransformerDecoder(config, self.embed,
                                             self.decode_vocab_size)
    logging.info("decode_vocab_size: {}".format(self.decode_vocab_size))
    logging.info("Initialize TransformerModel done.")

  def call(self, inputs, training=None, mask=None):
    input_enc_x = inputs["input_enc_x"]
    enc_mask = self.mask_layer(input_enc_x)
    enc_emb = self.embed(input_enc_x)
    enc_pos_emb = self.pos_embed(enc_emb)
    enc_emb = tf.keras.layers.add([enc_emb, enc_pos_emb])
    enc_emb = self.embed_d(enc_emb, training=training)
    enc_out = self.encoder(enc_emb, training=training, mask=enc_mask)

    if not self.is_infer:
      input_dec_x = inputs["input_dec_x"]
      dec_inputs = [input_dec_x, enc_out]
    else:
      dec_inputs = enc_out
    dec_out = self.decoder(dec_inputs, training=training, mask=enc_mask)
    return dec_out


@registers.model.register
class RnnSeq2SeqModel(Seq2SeqModel):
  """RNN model for text sequence to sequence"""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    logging.info("Initialize RnnSeq2SeqModel...")
    model_config = config['model']['net']['structure']
    self.is_infer = config['model']['is_infer']
    self.dropout_rate = model_config['dropout_rate']
    self.padding_token = utils.PAD_IDX
    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.encoder = layers.RnnEncoder(config, name="encoder")
    self.decoder = layers.RnnDecoder(
        config, self.decoder_embed, self.decode_vocab_size, name="decoder")
    self.mask_layer = tf.keras.layers.Lambda(lambda inputs: tf.cast(
        tf.not_equal(inputs, self.padding_token), tf.int32))

  def call(self, inputs, training=None, mask=None):
    enc_inputs = inputs["input_enc_x"]
    seq_enc_len = compute_sen_lens(enc_inputs, padding_token=self.padding_token)
    enc_mask = self.mask_layer(enc_inputs)
    enc_inputs = self.embed(enc_inputs)
    enc_inputs = self.embed_d(enc_inputs)
    enc_outputs, enc_state = self.encoder(
        enc_inputs, training=training, mask=enc_mask)
    if self.is_infer:
      dec_outputs = self.decoder([enc_outputs, enc_state, seq_enc_len],
                                 training=training)
      return dec_outputs

    else:
      dec_inputs = inputs["input_dec_x"]
      seq_dec_len = compute_sen_lens(
          dec_inputs, padding_token=self.padding_token)
      dec_outputs = self.decoder(
          [dec_inputs, seq_dec_len, enc_outputs, enc_state, seq_enc_len],
          training=training)
      return dec_outputs
