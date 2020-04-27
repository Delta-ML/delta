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
from delta.layers.dynamic_pooling import DynamicPoolingLayer
from delta.layers.match_pyramid import MatchingLayer
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
    # if self.tasktype == "Classification":
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


# pylint: disable=too-many-instance-attributes,too-many-ancestors
@registers.model.register
class MatchPyramidTextClassModel(MatchRnn):
  """Match texts model with Match Pyramid."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    logging.info("Initialize MatchPyramidTextClassModel ...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']
    self.max_seq_len = config['data']['task']['max_seq_len']
    model_config = config['model']['net']['structure']
    self.dropout_rate = model_config['dropout_rate']
    self.embedding_size = model_config['embedding_size']
    self.emb_trainable = model_config['emb_trainable']
    self.lstm_num_units = model_config['lstm_num_units']
    self.fc_num_units = model_config['fc_num_units']
    self.l2_reg_lambda = model_config['l2_reg_lambda']

    # Number of convolution blocks
    self.num_blocks = model_config['num_blocks']
    # The kernel count of the 2D convolution
    self.kernel_count = model_config['kernel_count']
    # The kernel size of the 2D convolution of each block
    self.kernel_size = model_config['kernel_size']
    # The max-pooling size of each block
    self.dpool_size = model_config['dpool_size']
    # The padding mode in the convolution layer
    self.padding = model_config['padding']
    # The activation function
    self.activation = model_config['activation']
    self.matching_type = model_config['matching_type']

    self.embed = tf.keras.layers.Embedding(
      self.vocab_size,
      self.embedding_size,
      trainable=self.emb_trainable,
      name='embdding',
      embeddings_initializer=self.embed_initializer)

    self.embed_d = tf.keras.layers.Dropout(self.dropout_rate)

    self.matching_layer = MatchingLayer(matching_type=self.matching_type)

    self.conv = []
    for i in range(self.num_blocks):
      conv = tf.keras.layers.Conv2D(
        self.kernel_count,
        self.kernel_size,
        padding=self.padding,
        activation=self.activation)
      self.conv.append(conv)

    self.dpool = DynamicPoolingLayer(*self.dpool_size)

    self.flatten = tf.keras.layers.Flatten()

    self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
    self.outlayer = tf.keras.layers.Dense(self.fc_num_units, activation='tanh')
    self.tasktype = config['data']['task']['type']
    # if self.tasktype == "Classification":
    self.final_dense = tf.keras.layers.Dense(
      self.num_classes,
      activation=tf.keras.activations.linear,
      name="final_dense")

    logging.info("Initialize MatchPyramidTextClassModel done.")

  def call(self, inputs, training=None, mask=None):  # pylint: disable=too-many-locals
    input_left = inputs["input_x_left"]
    input_right = inputs["input_x_right"]

    input_x_left_len = inputs["input_x_left_len"]
    input_x_right_len = inputs["input_x_right_len"]

    embedding = self.embed
    embed_left = embedding(input_left)
    embed_right = embedding(input_right)

    p_index = self._dynamic_pooling_index(input_x_left_len,
                                          input_x_right_len,
                                          self.max_seq_len,
                                          self.max_seq_len,
                                          1,
                                          1,
                                          )

    embed_cross = self.matching_layer([embed_left, embed_right])
    for i in range(self.num_blocks):
      embed_cross = self.conv[i](embed_cross)
    embed_pool = self.dpool(
      [embed_cross, p_index])

    embed_flat = self.flatten(embed_pool)

    dropout = self.dropout(embed_flat)
    out = self.outlayer(dropout)
    scores = self.final_dense(out)
    return scores



  def _dynamic_pooling_index(self, length_left,
                             length_right,
                             fixed_length_left: int,
                             fixed_length_right: int,
                             compress_ratio_left: float,
                             compress_ratio_right: float) -> tf.Tensor:
    def _dpool_index(one_length_left,
                     one_length_right,
                     fixed_length_left,
                     fixed_length_right):

      logging.info("fixed_length_left: {}".format(fixed_length_left))
      logging.info("fixed_length_right: {}".format(fixed_length_right))

      if one_length_left == 0:
        stride_left = fixed_length_left
      else:
        stride_left = 1.0 * fixed_length_left / tf.cast(one_length_left, dtype=tf.float32)

      if one_length_right == 0:
        stride_right = fixed_length_right
      else:
        stride_right = 1.0 * fixed_length_right / tf.cast(one_length_right, dtype=tf.float32)

      one_idx_left = [tf.cast(i / stride_left, dtype=tf.int32)
                      for i in range(fixed_length_left)]
      one_idx_right = [tf.cast(i / stride_right, dtype=tf.int32)
                       for i in range(fixed_length_right)]
      mesh1, mesh2 = tf.meshgrid(one_idx_left, one_idx_right)
      index_one = tf.transpose(
        tf.stack([mesh1, mesh2]), (2, 1, 0))
      return index_one

    index = []
    dpool_bias_left = dpool_bias_right = 0
    if fixed_length_left % compress_ratio_left != 0:
      dpool_bias_left = 1
    if fixed_length_right % compress_ratio_right != 0:
      dpool_bias_right = 1
    cur_fixed_length_left = int(
      fixed_length_left // compress_ratio_left) + dpool_bias_left
    cur_fixed_length_right = int(
      fixed_length_right // compress_ratio_right) + dpool_bias_right
    logging.info("length_left: {}".format(length_left))
    logging.info("length_right: {}".format(length_right))
    logging.info("cur_fixed_length_left: {}".format(cur_fixed_length_left))
    logging.info("cur_fixed_length_right: {}".format(cur_fixed_length_right))

    index = tf.map_fn(lambda x: _dpool_index(x[0], x[1], cur_fixed_length_left, cur_fixed_length_right),
                      (length_left, length_right), dtype=tf.int32)

    logging.info("index: {}".format(index))

    return index
