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
''' bilstmcrf model '''

import pickle
import delta.compat as tf
from absl import logging

from delta.models.text_cls_model import TextClassModel
from delta.utils.register import registers

# pylint: disable=too-many-ancestors, too-many-instance-attributes, abstract-method


class SeqclassModel(TextClassModel):
  """class for sequence labeling."""

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
class BilstmCrfModel(SeqclassModel):
  """bilstmcrf class for sequence labeling."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

    logging.info("Initialize BilstmModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']
    self.seq_num_classes = self.num_classes

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.num_layers = model_config['num_layers']
    self.l2_reg_lambda = model_config['l2_reg_lambda']
    self.batch_size = model_config['batch_size']
    self.max_len = model_config['max_len']
    self.num_units = model_config['num_units']
    self.fc_dim = model_config['fc_dim']  # output fully-connected layer size

    self.embed = tf.keras.layers.Embedding(
        input_dim=self.vocab_size,
        output_dim=self.embedding_size,
        mask_zero=True,
        input_length=self.max_len,
        embeddings_initializer=self.embed_initializer,
        trainable=True)

    self.embed_dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=self.num_units, return_sequences=True, recurrent_dropout=0.1),
        merge_mode='concat',
        name='BiLSTM')
    self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
    self.dense = tf.keras.layers.Dense(self.num_classes, name='fc_layer')

    self.transitions = self.add_weight(
        name='transitions', shape=(self.seq_num_classes, self.seq_num_classes))

    logging.info("Initialize BilstmCrfModel done.")

  def call(self, inputs, training=None, mask=None):
    input_x = inputs["input_x"]
    # [batch_size, max_len, embed_len]
    out = self.embed(input_x)
    # [batch_size, features]
    if self.use_pretrained_model:
      logging.info("use_pretrained_model: {}, {}".format(
          self.pretrained_model_name, self.pretrained_model_mode))
      if self.pretrained_model_name == 'bert' and \
        self.pretrained_model_mode == 'fine-tune':
        out = self.get_pre_train_graph(input_x)
      else:
        input_px = self.get_pre_train_graph(input_x)
        out = tf.concat([out, input_px], axis=-1)
        out = self.embed_dropout(out, training=training)
    out = self.bilstm(out)
    out = self.dropout(out, training=training)
    output = self.dense(out)
    return output
