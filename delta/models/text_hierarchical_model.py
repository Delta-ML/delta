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
"""Hierarchical text classification models."""

import pickle
from absl import logging
import delta.compat as tf

import delta.utils as utils
import delta.layers
from delta.layers.utils import cut_or_padding
from delta.layers.utils import compute_sen_lens
from delta.layers.utils import compute_doc_lens
from delta.layers.utils import split_one_doc_to_true_len_sens
from delta.models.text_cls_model import TextClassModel
from delta.utils.register import registers

# pylint: disable=abstract-method, too-many-ancestors, too-many-instance-attributes


@registers.model.register
class HierarchicalModel(TextClassModel):
  """Hierarchical text classification model."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

    logging.info("Initialize HierachicalModel...")

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

  @staticmethod
  def pad_to_hier_input(inputs, max_doc_len, max_sen_len, padding_token=0):
    """
    Input shape: [batch_size, max_len]
    New Input shape: [batch_size, max_doc_len, max_sen_len]
    """
    new_len = max_sen_len * max_doc_len
    new_input = cut_or_padding(inputs, new_len, padding_token=padding_token)
    new_input = tf.reshape(new_input, [-1, max_doc_len, max_sen_len])
    return new_input

  @staticmethod
  def pad_to_hier_input_true_len(inputs,
                                 max_doc_len,
                                 max_sen_len,
                                 split_token,
                                 padding_token=0):
    """
    Input shape: [batch_size, max_len]
    New Input shape: [batch_size, max_doc_len, max_sen_len]
    """
    new_input = tf.map_fn(
        lambda x: split_one_doc_to_true_len_sens(
            x, split_token, padding_token, max_doc_len, max_sen_len), inputs)
    return new_input


@registers.model.register
class HierarchicalAttentionModel(HierarchicalModel):
  """Hierarchical text classification model with attention."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

    logging.info("Initialize HierarchicalAttentionModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']
    self.use_true_length = config['model'].get('use_true_length', False)
    if self.use_true_length:
      self.split_token = config['data']['split_token']
    self.padding_token = utils.PAD_IDX

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.emb_trainable = model_config['emb_trainable']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.max_len = model_config['max_len']
    self.max_sen_len = model_config['max_sen_len']
    self.max_doc_len = model_config['max_doc_len']

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        trainable=self.emb_trainable,
        embeddings_initializer=self.embed_initializer)

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)
    self.sen_encoder = delta.layers.RnnAttentionEncoder(
        config, name="sen_encoder")
    self.doc_encoder = delta.layers.RnnAttentionEncoder(
        config, name="doc_encoder")

    self.final_dense = tf.keras.layers.Dense(
        self.num_classes,
        activation=tf.keras.activations.linear,
        name="final_dense")
    logging.info("Initialize HierarchicalAttentionModel done.")

  def call(self, inputs, training=None, mask=None):  # pylint: disable=too-many-locals
    input_x = tf.identity(inputs["input_x"], name='input_x')
    if self.use_dense_task:
      dense_input = inputs["input_dense"]
    if self.use_true_length:
      # [batch_size, max_doc_len, max_sen_len]
      input_hx = self.pad_to_hier_input_true_len(
          input_x,
          self.max_doc_len,
          self.max_sen_len,
          self.split_token,
          padding_token=self.padding_token)
    else:
      # [batch_size, max_doc_len, max_sen_len]
      input_hx = self.pad_to_hier_input(
          input_x,
          self.max_doc_len,
          self.max_sen_len,
          padding_token=self.padding_token)

    # [batch_size, max_doc_len]
    sen_lens = compute_sen_lens(input_hx, padding_token=self.padding_token)
    # [batch_size]
    doc_lens = compute_doc_lens(sen_lens)
    # [batch_size, max_doc_len, max_sen_len, 1]
    sen_mask = tf.expand_dims(
        tf.sequence_mask(sen_lens, self.max_sen_len, dtype=tf.float32), axis=-1)

    # [batch_size, max_doc_len, 1]
    doc_mask = tf.expand_dims(
        tf.sequence_mask(doc_lens, self.max_doc_len, dtype=tf.float32), axis=-1)

    # [batch_size, max_doc_len, max_sen_len, embed_len]
    out = self.embed(input_hx)
    if self.use_pretrained_model:
      input_px = self.get_pre_train_graph(input_x)
      input_px = tf.reshape(
          input_px,
          [-1, self.max_doc_len, self.max_sen_len, self.pretrained_model_dim])
      out = tf.concat([out, input_px], axis=-1)
    out = self.embed_d(out, training=training)
    all_sen_encoder = tf.keras.layers.TimeDistributed(self.sen_encoder)
    # [batch_size, max_doc_len, features]
    out = all_sen_encoder(out, training=training, mask=sen_mask)
    # [batch_size, features]
    out = self.doc_encoder(out, training=training, mask=doc_mask)

    if self.use_dense_input:
      dense_out = self.dense_input_linear(dense_input)
      if self.only_dense_input:
        out = dense_out
      else:
        out = tf.keras.layers.Concatenate()([out, dense_out])

    # [batch_size, class_num]
    scores = self.final_dense(out)

    return scores
