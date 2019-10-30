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
"""Sequence model for text classification."""

import pickle
import delta.compat as tf
from absl import logging

from delta import layers
from delta.models.text_cls_model import TextClassModel
from delta.utils.register import registers
from delta import utils

# pylint: disable=abstract-method, too-many-ancestors, too-many-instance-attributes


class SeqclassModel(TextClassModel):
  """Sequence model for text classification."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

    logging.info("Initialize SeqclassModel")

    self.use_pretrained_embedding = config['model']['use_pre_train_emb']
    if self.use_pretrained_embedding:
      self.embedding_path = config['model']['embedding_path']
      logging.info("Loading embedding file from: {}".format(
          self.embedding_path))
      self._word_embedding_init = pickle.load(open(self.embedding_path, 'rb'))
      self.embed_initializer = tf.constant_initializer(
          self._word_embedding_init)
    else:
      self.embed_initializer = tf.random_uniform_initializer(-0.1, 0.1)


@registers.model.register
class SeqclassCNNModel(SeqclassModel):
  """CNN model for text classification."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']

    self.sequence_length = config['data']['task']['max_seq_len']
    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']

    self.embedding_size = model_config['embedding_size']
    self.num_units = model_config['num_units']
    self.num_layers = model_config['num_layers']
    self.filter_sizes = model_config['filter_sizes']
    self.num_filters = model_config['num_filters']

    self.l2_reg_lambda = model_config['l2_reg_lambda']

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        embeddings_initializer=self.embed_initializer)

    self.conv2ds = []
    self.pools = []
    for i, filter_size in enumerate(self.filter_sizes):
      conv2d = tf.keras.layers.Conv2D(
          filters=self.num_filters,
          kernel_size=(filter_size, self.embedding_size),
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
          bias_initializer=tf.constant_initializer(value=0.0),
          padding='valid',
          name='conv_{}'.format(i))
      pool = tf.keras.layers.MaxPool2D(
          pool_size=(self.sequence_length - filter_size + 1, 1),
          strides=(1, 1),
          padding='valid',
          name='name_{}'.format(i))
      self.conv2ds.append(conv2d)
      self.pools.append(pool)

    self.flat = tf.keras.layers.Flatten()

    self.dense = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)

    self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    self.final_dense = tf.keras.layers.Dense(
        self.num_classes, activation=tf.keras.activations.linear)

  def call(self, inputs, training=None, mask=None):
    input_x = tf.identity(inputs["input_x"], name="input_x")
    if self.use_dense_task:
      dense_input = inputs["input_dense"]
    embed = self.embed(input_x)
    embed_expand = tf.expand_dims(embed, axis=-1)
    conv_outs = [conv2d(embed_expand) for conv2d in self.conv2ds]
    pool_outs = [pool(co) for co, pool in zip(conv_outs, self.pools)]
    out = tf.keras.layers.Concatenate(axis=1)(pool_outs)
    out = self.flat(out)
    out = self.dropout(out, training=training)
    out = self.dense(out)
    if self.use_dense_input:
      dense_out = self.dense_input_linear(dense_input)
      if self.only_dense_input:
        out = dense_out
      else:
        out = tf.keras.layers.Concatenate()([out, dense_out])
    scores = self.final_dense(out)
    return scores


@registers.model.register
class RnnAttentionModel(SeqclassModel):
  """RNN model for text classification."""

  def __init__(self, config, **kwargs):
    super(RnnAttentionModel, self).__init__(config, **kwargs)

    logging.info("Initialize RnnAttentionModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.batch_size = model_config['batch_size']
    self.max_len = model_config['max_len']

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        embeddings_initializer=self.embed_initializer)

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.encoder = layers.RnnAttentionEncoder(config, name="encoder")
    self.final_dense = tf.keras.layers.Dense(
        self.num_classes,
        activation=tf.keras.activations.linear,
        name="final_dense")
    logging.info("Initialize RnnAttentionModel done.")

  @staticmethod
  def compute_lens(inputs, max_len):
    """count sequence length.
    input: [batch_size, max_len]
    lens: [batch_size]
    """

    x_binary = tf.cast(tf.cast(tf.reverse(inputs, axis=[1]), tf.bool), tf.int32)
    lens = max_len - tf.argmax(x_binary, axis=1, output_type=tf.int32)

    zeros = tf.zeros_like(lens, dtype=tf.int32)
    x_sum = tf.reduce_sum(inputs, axis=1)
    sen_lens = tf.where(tf.equal(x_sum, 0), zeros, lens)
    return sen_lens

  def call(self, inputs, training=None, mask=None):
    input_x = inputs["input_x"]
    if self.use_dense_task:
      dense_input = inputs["input_dense"]

    # [batch_size]
    lens = self.compute_lens(input_x, self.max_len)

    # [batch_size, max_len, 1]
    mask = tf.expand_dims(
        tf.sequence_mask(lens, self.max_len, dtype=tf.float32), axis=-1)

    # [batch_size, max_len, embed_len]
    out = self.embed(input_x)
    out = self.embed_d(out, training=training)
    # [batch_size, features]
    out = self.encoder(out, training=training, mask=mask)
    if self.use_dense_input:
      dense_out = self.dense_input_linear(dense_input)
      if self.only_dense_input:
        out = dense_out
      else:
        out = tf.keras.layers.Concatenate()([out, dense_out])
    # [batch_size, class_num]
    scores = self.final_dense(out)
    return scores


@registers.model.register
class TransformerModel(SeqclassModel):
  """Transformer model for text classification"""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    logging.info("Initialize TransformerModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.max_len = model_config['max_len']
    self.transformer_dropout = model_config['transformer_dropout']
    self.residual_conn = model_config['residual_conn']
    self.head_num = model_config['head_num']
    self.hidden_dim = model_config['hidden_dim']
    self.padding_token = utils.PAD_IDX

    self.mask_layer = tf.keras.layers.Lambda(lambda inputs: tf.cast(
        tf.not_equal(inputs, self.padding_token), tf.int32))
    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        embeddings_initializer=self.embed_initializer)

    self.pos_embed = layers.PositionEmbedding(self.max_len, self.embedding_size)

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)

    self.transformer_encoder = layers.TransformerEncoder(config)

    self.pool = tf.keras.layers.GlobalMaxPooling1D()

    self.final_dense = tf.keras.layers.Dense(
        self.num_classes,
        activation=tf.keras.activations.linear,
        name="final_dense")
    logging.info("Initialize TransformerModel done.")

  def call(self, inputs, training=None, mask=None):
    input_x = inputs["input_x"]
    if self.use_dense_task:
      dense_input = inputs["input_dense"]

    enc_mask = self.mask_layer(input_x)
    emb = self.embed(input_x)
    pos_emb = self.pos_embed(emb)
    emb = tf.keras.layers.add([emb, pos_emb])
    enc_emb = self.embed_d(emb, training=training)

    enc_out = self.encoder(enc_emb, training=training, mask=enc_mask)

    out = self.pool(enc_out)

    if self.use_dense_input:
      dense_out = self.dense_input_linear(dense_input)
      if self.only_dense_input:
        out = dense_out
      else:
        out = tf.keras.layers.Concatenate()([out, dense_out])

    scores = self.final_dense(out)
    return scores


@registers.model.register
class FullyConnectModel(SeqclassModel):
  """FullyConnect model for text classification based on
  pretrain embedding/model"""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    logging.info("Initialize FullyConnectModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.max_len = model_config['max_len']

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        embeddings_initializer=self.embed_initializer)

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.final_dense = tf.keras.layers.Dense(
        self.num_classes,
        activation=tf.keras.activations.linear,
        name="final_dense")
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    logging.info("Initialize FullyConnectModel done.")

  def call(self, inputs, training=None, mask=None):
    input_x = inputs["input_x"]
    if self.use_dense_task:
      dense_input = inputs["input_dense"]

    # [batch_size, max_len, embed_len]
    out = self.embed(input_x)
    if self.use_pretrained_model:
      logging.info("use_pretrained_model: {}, {}".format(
          self.pretrained_model_name, self.pretrained_model_mode))
      if self.pretrained_model_name == 'elmo':
        input_px = self.get_pre_train_graph(input_x)
        input_px = tf.reshape(input_px,
                              [-1, self.max_len, self.pretrained_model_dim])
        out = tf.concat([out, input_px], axis=-1)
        out = tf.reduce_max(out, axis=1)
      if self.pretrained_model_name == 'bert':
        out = self.get_pre_train_graph(input_x)
    else:
      out = tf.reduce_max(out, axis=1)
    out = self.embed_d(out, training=training)
    if self.use_dense_input:
      dense_out = self.dense_input_linear(dense_input)
      if self.only_dense_input:
        out = dense_out
      else:
        out = tf.keras.layers.Concatenate()([out, dense_out])
    # [batch_size, class_num]
    scores = self.final_dense(out)
    return scores
