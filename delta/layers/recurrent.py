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

import tensorflow as tf

import delta
from delta.layers.base_layer import Layer


class RnnAttentionEncoder(Layer):  # pylint: disable=too-many-instance-attributes
  """
  RNN + Attention
  Input Shape: [batch_size, steps, features]
  Output Shape: [batch_size, units]
  """

  def __init__(self, config, **kwargs):
    super(RnnAttentionEncoder, self).__init__(**kwargs)
    tf.logging.info("Initialize RnnAttentionEncoder {}...".format(self.name))

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
      tf.logging.error(error_info)
      raise ValueError(error_info)

    self.sen_encoder = tf.keras.layers.Bidirectional(
        rnn_class(self.cell_dim, return_sequences=True))
    # self.sen_all_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200))
    self.sen_att = delta.layers.HanAttention(name="{}_att".format(self.name))
    self.sen_att_d = tf.keras.layers.Dropout(self.dropout_rate)
    # self.sen_att_d_bn = tf.keras.layers.BatchNormalization()
    tf.logging.info("Initialize RnnAttentionEncoder {} Done.".format(self.name))

  def build(self, input_shape):
    self.built = True

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([input_shape[0], self.cell_dim * 2])

  def call(self, inputs, training=None, mask=None):
    out = self.sen_encoder(inputs)
    # out = self.sen_all_dense(out)
    out = self.sen_att(out, mask=mask)
    out = self.sen_att_d(out, training=training)
    # out = self.sen_att_d_bn(out, training=training)
    return out
