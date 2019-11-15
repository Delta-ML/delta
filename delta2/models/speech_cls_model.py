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
''' emotion keras model'''
import numpy as np
from absl import logging

#pylint: disable=no-name-in-module
import delta.compat as tf
from tensorflow.python.keras import backend as K

from delta.models.base_model import Model
from delta.layers.base_layer import Layer

from delta import utils
from delta.utils.register import registers

#pylint: disable=invalid-name
#pylint: disable=attribute-defined-outside-init
#pylint: disable=missing-docstring
#pylint: disable=too-many-instance-attributes
#pylint: disable=attribute-defined-outside-init
#pylint: disable=too-many-ancestors

layers = tf.keras.layers


@registers.model.register
class EmoLstmModel(Model):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.config = config

  def build(self, input_shape):
    logging.info(f"{self.__class__.__name__} input_shape : {input_shape}")
    _, time, feat, channels = input_shape['inputs'].as_list()

    self.reshape = layers.Reshape((time, feat * channels),
                                  input_shape=(time, feat, channels))
    self.lstm1 = layers.LSTM(512, return_sequences=True)
    self.lstm2 = layers.LSTM(256, return_sequences=False)
    self.dense1 = layers.Dense(512, activation='relu')
    self.drop1 = layers.Dropout(rate=0.2)
    self.dense2 = layers.Dense(4)

    # https://stackoverflow.com/questions/55684949/subclass-of-tf-keras-model-can-not-get-summay-result
    # https://stackoverflow.com/questions/52826134/keras-model-subclassing-examples
    x = {}
    for key, shape in input_shape.items():
      x[key] = tf.convert_to_tensor(
          np.random.normal(size=[1] + shape.as_list()[1:]),
          dtype=tf.keras.backend.floatx())
    _ = self.call(x)
    #super().build(input_shape=[input_shape['inputs'].as_list(), input_shape['labels'].as_list()])
    self.built = True

  def call(self, inputs, training=None, mask=None):
    logging.info(f"xxxx input: {inputs}, training: {training}")
    if isinstance(inputs, dict):
      x = inputs['inputs']
    elif isinstance(inputs, list):
      x = inputs[0]
    else:
      x = inputs
    x = self.reshape(x)
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.dense1(x)
    #x = self.drop1(x, training=training)
    logits = self.dense2(x)
    return logits


@registers.model.register
class EmoBLstmModel(Model):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.config = config

  def build(self, input_shape):
    logging.info(f"{self.__class__.__name__} input_shape : {input_shape}")
    _, time, feat, channels = input_shape['inputs'].as_list()

    self.reshape = layers.Reshape((time, feat * channels),
                                  input_shape=(time, feat, channels))
    self.lstm1 = layers.Bidirectional(layers.LSTM(512, return_sequences=True))
    self.lstm2 = layers.Bidirectional(layers.LSTM(256, return_sequences=False))
    self.dense1 = layers.Dense(512, activation='relu')
    self.dense2 = layers.Dense(4)
    self.drop1 = layers.Dropout(rate=0.2)

    self.built = True

  def call(self, inputs, training=None, mask=None):
    logging.info(f"xxxx input: {inputs}, training: {training}")
    x = inputs['inputs']
    x = self.reshape(x)
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.dense1(x)
    x = self.drop1(x, training=training)
    logits = self.dense2(x)
    return logits


@registers.model.register
class EmoBLstmPoolModel(Model):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.config = config

  def build(self, input_shape):
    logging.info(f"{self.__class__.__name__} input_shape : {input_shape}")
    _, time, feat, channels = input_shape['inputs'].as_list()

    self.reshape = layers.Reshape((time, feat * channels),
                                  input_shape=(time, feat, channels))
    self.dense1 = layers.TimeDistributed(layers.Dense(512, activation='relu'))
    self.drop1 = layers.Dropout(0.5)
    self.lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
    self.drop2 = layers.Dropout(0.5)
    self.avg_pool = layers.GlobalAveragePooling1D()
    self.drop3 = layers.Dropout(rate=0.5)
    self.dense2 = layers.Dense(4)

    self.built = True

  def call(self, inputs, training=None, mask=None):
    logging.info(f"xxxx input: {inputs}, training: {training}")
    x = inputs['inputs']
    x = self.reshape(x)
    x = self.dense1(x)
    x = self.drop1(x, training=training)
    x = self.lstm1(x)
    x = self.drop2(x, training=training)
    x = self.avg_pool(x)
    x = self.drop3(x, training=training)
    logits = self.dense2(x)
    return logits


#pylint: disable=too-many-instance-attributes
class CBDP(Layer):

  #pylint: disable=too-many-arguments
  def __init__(self,
               filters=128,
               filter_size=(5, 3),
               filter_strides=(1, 1),
               pool_size=(4, 4),
               drop_rate=0.1):
    super().__init__(name='cbdp')
    self.filters = filters
    self.filter_size = filter_size
    self.filter_strides = filter_strides
    self.pool_size = pool_size
    self.drop_rate = drop_rate

    self.conv = tf.keras.layers.Conv2D(
        filters,
        filter_size,
        strides=filter_strides,
        kernel_initializer='glorot_normal',
        use_bias=True,
        padding='same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.drop = tf.keras.layers.Dropout(self.drop_rate)
    self.pool = tf.keras.layers.MaxPool2D(
        pool_size=self.pool_size,
        strides=self.pool_size,
        padding='same',
        data_format='channels_last')

  #pylint: disable=arguments-differ
  def call(self, x, training):
    ''' x shape: [batch, frame, feat, channel]'''
    x = self.conv(x)
    x = self.bn(x, training=training)
    x = tf.nn.relu6(x)
    x = self.drop(x, training=training)
    x = self.pool(x)
    return x


class CNN(Layer):

  def __init__(self, drop_rate):
    super().__init__(name='cnn')
    self.drop_rate = drop_rate

    self.block1 = CBDP(
        filters=128,
        filter_size=[5, 3],
        filter_strides=(1, 1),
        pool_size=(4, 4),
        drop_rate=self.drop_rate)
    self.block2 = CBDP(
        filters=256,
        filter_size=[5, 3],
        filter_strides=(1, 1),
        pool_size=(1, 2),
        drop_rate=self.drop_rate)
    self.block3 = CBDP(
        filters=256,
        filter_size=[5, 3],
        filter_strides=(1, 1),
        pool_size=(1, 2),
        drop_rate=self.drop_rate)

  #pylint: disable=arguments-differ
  def call(self, x, training):
    x = self.block1(x, training)
    x = self.block2(x, training)
    x = self.block3(x, training)
    return x


class Linear(Layer):
  ''' linear layer'''

  def __init__(self, dense_dim, drop_rate):
    super().__init__(name='linear')
    self.drop_rate = drop_rate
    self.fc_dim = dense_dim

    self.fc = tf.keras.layers.Dense(self.fc_dim)
    self.drop = tf.keras.layers.Dropout(self.drop_rate)

  def build(self, input_shape):
    batch, time, feat, channl = input_shape
    del batch
    self.reshape1 = tf.keras.layers.Reshape((time, feat * channl))
    #self.reshape2 = tf.keras.layers.Reshape((time, self.fc_dim))

  #pylint: disable=arguments-differ
  def call(self, x, training):
    ''' x shape: [batch, time, feat, channel]
        output shape: [batch, time, dim ]
    '''
    x = self.reshape1(x)
    x = self.fc(x)
    x = tf.nn.relu6(x)
    x = self.drop(x, training=training)
    #x = self.reshape2(x)
    return x


class TimePool(Layer):

  def __init__(self):
    super().__init__(name='time_pool')

  def build(self, input_shape):
    time, dim = input_shape[1:]
    self.time_pool = tf.keras.layers.AveragePooling1D(
        pool_size=time.value, strides=time.value, padding='same')
    self.reshape = tf.keras.layers.Reshape((dim,))

  #pylint: disable=arguments-differ
  def call(self, x):
    '''
      x shape :[ batch, time, dim]
      output shape: [batch, dim]
    '''
    x = self.time_pool(x)
    x = self.reshape(x)
    return x


class LBD(Layer):

  def __init__(self, dense_dim, drop_rate):
    super().__init__(name='lbd')
    self.dense_dim = dense_dim
    self.drop_rate = drop_rate

    self.fc = tf.keras.layers.Dense(self.dense_dim)
    self.drop = tf.keras.layers.Dropout(self.drop_rate)
    self.bn = tf.keras.layers.BatchNormalization()

  #pylint: disable=arguments-differ
  def call(self, x, training):
    x = self.fc(x)
    x = self.bn(x, training)
    x = tf.nn.relu6(x)
    x = self.drop(x, training)
    return x


class Head(Layer):
  ''' output logits '''

  def __init__(self, num_class):
    super().__init__(name='head')
    self.num_class = num_class

    self.fc = tf.keras.layers.Dense(self.num_class)

  #pylint: disable=arguments-differ
  def call(self, x):
    x = self.fc(x)
    return x


class Feat(Layer):

  def __init__(self, cmvn_path):
    super().__init__(name='cmvn', trainable=False)
    self.mean, self.std = utils.load_cmvn(cmvn_path)

  #pylint: disable=arguments-differ
  def call(self, x):
    x = utils.apply_cmvn(x, self.mean, self.std)
    return x


#pylint: disable=too-many-instance-attributes,too-many-ancestors
@registers.model.register
class EmoCRNNModel(Model):
  ''' main model '''

  def __init__(self, drop_rate):
    super().__init__()
    self.drop_rate = drop_rate
    self.fc1_dim = 786
    self.fc2_dim = 64
    self.fc3_dim = 2

    self.cnn = CNN(self.drop_rate)
    self.fc1 = Linear(self.fc1_dim, self.drop_rate)
    self.time_pool = TimePool()
    self.fc2 = LBD(self.fc2_dim, self.drop_rate)
    self.head = Head(self.fc3_dim)

  #pylint: disable=arguments-differ
  def call(self, features, training):
    x = features['inputs']
    x = self.cnn(x, training)
    x = self.fc1(x, training)
    x = self.time_pool(x)
    x = self.fc2(x, training)
    x = self.head(x)
    return x


#pylint: disable=too-many-instance-attributes,too-many-ancestors
@registers.model.register
class EmoCFNNModel(Model):
  ''' main model '''

  #pylint: disable=useless-super-delegation
  def __init__(self):
    super().__init__()

  def build(self, input_shape):
    batch, time, feat, channels = input_shape.as_list()
    del batch
    stride = 3
    self.cnn1 = tf.keras.layers.Conv2D(
        filters=channels * 2,
        kernel_size=(5, 3),
        strides=(2, 1),
        padding='same',
        use_bias=True)
    self.cnn2 = tf.keras.layers.Conv2D(
        filters=channels * 2,
        kernel_size=(5, 3),
        strides=(stride, 1),
        padding='same',
        use_bias=True)
    self.cnn3 = tf.keras.layers.Conv2D(
        filters=channels * 2,
        kernel_size=(5, 3),
        strides=(2, 1),
        padding='same',
        use_bias=True)
    time /= (2 * stride * 2)

    def tanh_sig(x):
      x1, x2 = tf.split(x, 2, axis=-1)
      return tf.nn.tanh(x1) * tf.nn.sigmoid(x2)

    #pylint: disable=unnecessary-lambda
    self.tanh_sig = tf.keras.layers.Lambda(lambda x: tanh_sig(x))

    self.reshape = tf.keras.layers.Reshape((int(time), int(feat * channels)))
    self.cnn4 = tf.keras.layers.Conv1D(
        filters=int(feat * channels / 2),
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=True)
    self.cnn5 = tf.keras.layers.Conv1D(
        filters=(feat * channels),
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=True)
    self.cnn6 = tf.keras.layers.Conv1D(
        filters=(feat * channels),
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True)

    self.cnn7 = tf.keras.layers.Conv1D(
        filters=int(feat * channels / 2),
        kernel_size=5,
        strides=2,
        padding='same',
        use_bias=True)
    time /= 2
    self.reshape2 = tf.keras.layers.Reshape((int(time * feat * channels / 2),))

    self.nn1 = tf.keras.layers.Dense(3000)
    self.reshape3 = tf.keras.layers.Reshape(
        (int(3000 / (feat * channels / 2)), int(feat * channels / 2)))

    self.max_pool = tf.keras.layers.GlobalMaxPool1D()
    self.nn2 = tf.keras.layers.Dense(2)

  #pylint: disable=arguments-differ
  def call(self, features):
    x = features['inputs']
    x = self.cnn1(x)
    x = self.tanh_sig(x)
    x = self.cnn2(x)
    x = self.tanh_sig(x)
    x = self.cnn3(x)
    x1 = self.tanh_sig(x)

    x2 = self.reshape(x1)

    x3 = self.cnn4(x2)
    x3 = tf.nn.relu(x3)
    x3 = self.cnn5(x2)
    x3 = tf.nn.relu(x3)
    x3 += x2

    x4 = self.cnn6(x2)
    x4 = tf.nn.relu(x4)
    x4 += x2

    x5 = x3 + x4
    x6 = self.cnn7(x5)
    x6 = tf.nn.relu(x6)
    x6 = self.reshape2(x6)

    x6 = self.nn1(x6)
    x6 = tf.nn.relu(x6)
    x7 = self.reshape3(x6)

    x8 = self.max_pool(x7)
    logits = self.nn2(x8)
    return logits
