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
"""Match texts with Rnn models."""

import pickle
from absl import logging
import delta.compat as tf
from delta.models.base_model import Model
from delta.utils.register import registers


# pylint: disable=too-few-public-methods, abstract-method,too-many-ancestors
@registers.model.register
class MatchRnn(Model):
  """Match texts with Rnn models."""

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)

    logging.info("Initialize MatchRnn...")

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


# pylint: disable=too-many-instance-attributes,too-many-ancestors
@registers.model.register
class MatchRnnTextClassModel(MatchRnn):
  """Match texts model with Rnn and Attention."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    logging.info("Initialize MatchRnnTextClassModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']

    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.emb_trainable = model_config['emb_trainable']
    self.lstm_num_units = model_config['lstm_num_units']
    self.fc_num_units = model_config['fc_num_units']
    self.l2_reg_lambda = model_config['l2_reg_lambda']

    self.embed = tf.keras.layers.Embedding(
        self.vocab_size,
        self.embedding_size,
        trainable=self.emb_trainable,
        name='embdding',
        embeddings_initializer=self.embed_initializer)

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)

    self.lstm_left = tf.keras.layers.LSTM(
        self.lstm_num_units, return_sequences=True, name='lstm_left')
    self.lstm_right = tf.keras.layers.LSTM(
        self.lstm_num_units, return_sequences=True, name='lstm_right')
    self.concat = tf.keras.layers.Concatenate(axis=1)

    self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.outlayer = tf.keras.layers.Dense(self.fc_num_units, activation='tanh')
    self.tasktype = config['data']['task']['type']
    #if self.tasktype == "Classification":
    self.final_dense = tf.keras.layers.Dense(
        self.num_classes,
        activation=tf.keras.activations.linear,
        name="final_dense")

    logging.info("Initialize MatchRnnTextClassModel done.")

  def call(self, inputs, training=None, mask=None):  # pylint: disable=too-many-locals

    input_left = inputs["input_x_left"]
    input_right = inputs["input_x_right"]

    embedding = self.embed
    embed_left = embedding(input_left)
    embed_right = embedding(input_right)

    encoded_left = self.lstm_left(embed_left)
    encoded_right = self.lstm_right(embed_right)

    encoded_right = tf.transpose(encoded_right, [0, 2, 1])
    left_right_sim = tf.matmul(encoded_left, encoded_right)
    shape_list = left_right_sim.get_shape()
    newdim = shape_list[1] * shape_list[2]
    sim_matrix = tf.reshape(left_right_sim, [-1, newdim], name="sim_matrix")

    dropout = self.dropout(sim_matrix)
    out = self.outlayer(dropout)

    scores = self.final_dense(out)

    return scores
