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
''' asr ctc model '''
import delta.compat as tf
#pylint: disable=import-error,unused-import
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input

from absl import logging

#delta
from delta import utils
from delta.utils.loss.loss_impl import CTCLoss as ctc_loss
from delta.models.base_model import RawModel
from delta.utils.register import registers

#pylint: disable=invalid-name,missing-docstring


@registers.model.register
class CTCAsrModel(RawModel):
  '''
  CTC ASR Model
  reference: https://github.com/holm-aune-bachelor2018/ctc
  '''

  def __init__(self, config, name=None):
    super().__init__(name=name)
    self._config = config

    logging.info("--- dummy Task to get meta data ---")
    logging.info("--- do not care the Task mode here ---")
    task = utils.task(config, mode=utils.TRAIN)
    logging.info("--- dummy Task to get meta data ---")
    logging.flush()

    self._feat_shape = task.feat_shape
    self._vocab_size = task.vocab_size

    self.build()

  @property
  def feat_shape(self):
    assert isinstance(self._feat_shape, (list))
    return self._feat_shape

  @property
  def config(self):
    return self._config

  def get_loss_fn(self):
    return ctc_loss(self._config)
    #return utils.loss(self._config)

  def ctc_lambda_func(self, args):
    y_pred, input_length, labels, label_length = args
    return self.get_loss_fn()(
        logits=y_pred,
        input_length=input_length,
        labels=labels,
        label_length=label_length,
        name='ctc_loss')
    #return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

  def build(self):
    input_tensor = Input(
        name='inputs', shape=(None, *self._feat_shape, 1), dtype=tf.float32)

    x = input_tensor

    x = Conv2D(
        filters=32,
        kernel_size=(11, 5),
        use_bias=True,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        name="conv1")(
            x)

    x = Conv2D(
        filters=32,
        kernel_size=(11, 5),
        use_bias=True,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        name="conv2")(
            x)

    _, _, dim, channels = x.get_shape().as_list()
    output_dim = dim * channels
    x = Reshape((-1, output_dim))(x)

    x = TimeDistributed(Dropout(0.2))(x)
    x = Bidirectional(
        LSTM(
            units=512,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm'))(
                x)

    x = TimeDistributed(Dropout(0.2))(x)
    x = Bidirectional(
        LSTM(
            512,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm1'))(
                x)

    x = TimeDistributed(Dropout(0.2))(x)
    x = Bidirectional(
        LSTM(
            512,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm2'))(
                x)

    x = TimeDistributed(Dropout(0.2))(x)
    x = Bidirectional(
        LSTM(
            512,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm3'))(
                x)

    x = TimeDistributed(Dense(1024, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    # Output layer with softmax
    x = TimeDistributed(Dense(self._vocab_size), name="outputs")(x)

    input_length = Input(name='input_length', shape=[], dtype='int64')
    labels = Input(name='targets', shape=[None], dtype='int32')
    label_length = Input(name='target_length', shape=[], dtype='int64')
    loss_out = Lambda(
        self.ctc_lambda_func, output_shape=(),
        name='ctc')([x, input_length, labels, label_length])

    self._model = tf.keras.Model(
        inputs=[input_tensor, labels, input_length, label_length],
        outputs=[loss_out])

  @property
  def model(self):
    return self._model

  def call(self, inputs, **kwargs):
    output = self.model(inputs, **kwargs)
    return output


@registers.model.register
class CTC5BlstmAsrModel(CTCAsrModel):
  '''
  CTC ASR Model
  reference: https://www.cs.cmu.edu/~ymiao/pub/icassp2016_ctc.pdf
  '''

  def build(self):
    input_tensor = Input(
        name='inputs', shape=(None, *self._feat_shape, 1), dtype=tf.float32)

    x = input_tensor
    _, _, dim, channels = x.get_shape().as_list()
    output_dim = dim * channels
    x = Reshape((-1, output_dim))(x)

    x = Bidirectional(
        LSTM(
            units=320,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm'))(
                x)

    x = Bidirectional(
        LSTM(
            units=320,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm1'))(
                x)

    x = Bidirectional(
        LSTM(
            units=320,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm2'))(
                x)

    x = Bidirectional(
        LSTM(
            units=320,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm3'))(
                x)

    x = Bidirectional(
        LSTM(
            units=320,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            return_sequences=True,
            name='lstm4'))(
                x)

    # Output layer with softmax
    x = TimeDistributed(Dense(self._vocab_size))(x)

    input_length = Input(name='input_length', shape=[], dtype='int64')
    labels = Input(name='targets', shape=[None], dtype='int32')
    label_length = Input(name='target_length', shape=[], dtype='int64')
    loss_out = Lambda(
        self.ctc_lambda_func, output_shape=(),
        name='ctc')([x, input_length, labels, label_length])

    self._model = tf.keras.Model(
        inputs=[input_tensor, labels, input_length, label_length],
        outputs=[loss_out])
