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
import tensorflow as tf
#pylint: disable=import-error,unused-import
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Activation, CuDNNLSTM, TimeDistributed, Dense, Reshape
from tensorflow.keras.layers import Lambda, Input

#delta
from delta import utils
from delta.models.asr_model import CTCAsrModel
from delta.utils.register import registers

#pylint: disable=invalid-name,missing-docstring


@registers.model.register
class CTCRefAsrModel(CTCAsrModel):
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
        320,
        kernel_initializer='glorot_uniform',
        bias_initializer='random_normal',
        return_sequences=True,
        name='lstm'))(
            x)
    x = Activation('tanh', name='activation')(x)

    x = Bidirectional(
        LSTM(
        320,
        kernel_initializer='glorot_uniform',
        bias_initializer='random_normal',
        return_sequences=True,
        name='lstm1'))(
            x)  
    x = Activation('tanh', name='activation1')(x)

   
    x = Bidirectional(
        LSTM(
        320,
        kernel_initializer='glorot_uniform',
        bias_initializer='random_normal',
        return_sequences=True,
        name='lstm2'))(
            x)  
    x = Activation('tanh', name='activation2')(x)
   
    x = Bidirectional(
        LSTM(
        320,
        kernel_initializer='glorot_uniform',
        bias_initializer='random_normal',
        return_sequences=True,
        name='lstm3'))(
            x)  
    x = Activation('tanh', name='activation3')(x)

    x = Bidirectional(
        LSTM(
        320,
        kernel_initializer='glorot_uniform',
        bias_initializer='random_normal',
        return_sequences=True,
        name='lstm4'))(
            x)  
    x = Activation('tanh', name='activation4')(x)

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
