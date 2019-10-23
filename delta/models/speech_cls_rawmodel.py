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
''' emotion crnn model support multi-modal'''
from absl import logging
import delta.compat as tf

from delta import utils
from delta.layers import common_layers
from delta.data.feat.tf_speech_feature import speech_params, extract_feature

from delta.models.base_model import RawModel
from delta.utils.register import registers

#pylint: disable=invalid-name


#pylint: disable=too-many-instance-attributes
@registers.model.register
class EmoCRNNRawModel(RawModel):
  ''' main model '''

  def __init__(self, config, name=None):
    super().__init__(name=name)
    self.config = config

    self.netconf = self.config['model']['net']['structure']
    self.taskconf = self.config['data']['task']
    self.audioconf = self.taskconf['audio']

    self.attention = self.netconf['attention']
    self.vocab_size = self.taskconf['text']['vocab_size']
    frame_per_sec = 1 / self.taskconf['audio']['winstep']
    self.input_len = self.taskconf['audio']['clip_size'] * frame_per_sec
    self.input_type = 'samples' if self.taskconf[
        'suffix'] == '.wav' else 'features'
    self.input_channels = 3 if self.taskconf['audio']['add_delta_deltas'] else 1

    # l2
    self._extra_train_ops = []

    self.std = None
    self.hp = None
    self.alphas = None
    self.train = None
    self.mean = None

  def preprocess(self, inputs, input_text):
    ''' preprocess speech and text inputs
    params:
      inputs: speech input
      input_text: text input
    '''
    with tf.variable_scope('feature'):
      if self.input_type == 'samples':
        # speech feature config
        self.hp = speech_params(
            sr=self.taskconf['audio']['sr'],
            bins=self.audioconf['feature_size'],
            dither=self.train,
            use_delta_deltas=self.audioconf['add_delta_deltas'],
            cmvn=self.audioconf['cmvn'],
            cmvn_path=self.audioconf['cmvn_path'])

        feats = extract_feature(inputs, params=self.hp)
      else:
        self.mean, self.std = utils.load_cmvn(self.audioconf['cmvn_path'])
        feats = utils.apply_cmvn(inputs, self.mean, self.std)
    return feats, input_text

  #pylint: disable=arguments-differ
  def call(self, features, **kwargs):
    self.train = kwargs['training']
    feats = tf.identity(features['inputs'], name='feats')
    texts = features['texts']

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      feats, texts = self.preprocess(feats, texts)
      logits = self.model(feats, texts)
    return logits

  #pylint: disable=too-many-locals
  def conv_block(self, inputs, depthwise=False):
    ''' conv layers'''
    filters = self.netconf['filters']
    logging.info("filters : {}".format(filters))
    filters_size = self.netconf['filter_size']
    logging.info("filters_size : {}".format(filters_size))
    filters_strides = self.netconf['filter_stride']
    logging.info("filters_strides : {}".format(filters_strides))
    pools_size = self.netconf['pool_size']
    logging.info("pools_size : {}".format(pools_size))

    layer_num = len(filters)
    assert layer_num == len(filters_size)
    assert layer_num == len(filters_strides)
    assert layer_num == len(pools_size)

    channels = [self.input_channels] + filters
    logging.info("channels : {}".format(channels))

    downsample_input_len = self.input_len
    with tf.variable_scope('cnn'):
      x = tf.identity(inputs)
      for index, _ in enumerate(filters):
        unit_name = 'unit-' + str(index + 1)
        with tf.variable_scope(unit_name):
          if depthwise:
            x = tf.layers.separable_conv2d(
                x,
                filters=filters[index],
                kernel_size=filters_size[index],
                strides=filters_strides[index],
                padding='same',
                name=unit_name)
          else:
            cnn_name = 'cnn-' + str(index + 1)
            x = common_layers.conv2d(x, cnn_name, filters_size[index],
                                     channels[index], channels[index + 1],
                                     filters_strides[index])
          if self.netconf['use_bn']:
            bn_name = 'bn' + str(index + 1)
            x = tf.layers.batch_normalization(
                x, axis=-1, momentum=0.9, training=self.train, name=bn_name)
          x = tf.nn.relu6(x)
          if self.netconf['use_dropout']:
            x = tf.layers.dropout(
                x, self.netconf['dropout_rate'], training=self.train)
          x = common_layers.max_pool(x, pools_size[index], pools_size[index])
          downsample_input_len = downsample_input_len / pools_size[index][0]

    return x, downsample_input_len

  def linear_block(self, x):
    '''
    linear layer for dim reduction
    x: shape [batch, time, feat, channel]
    output: shape [b, t, f]
    '''
    with tf.variable_scope('linear'):
      times, feat, channel = x.shape.as_list()[1:]
      x = tf.reshape(x, [-1, feat * channel])
      if self.netconf['use_dropout']:
        x = tf.layers.dropout(
            x, self.netconf['dropout_rate'], training=self.train)
      x = common_layers.linear(x, 'linear1',
                               [feat * channel, self.netconf['linear_num']])
      #x = tf.nn.relu6(x)
      x = tf.reshape(x, [-1, times, self.netconf['linear_num']])
    return x

  def lstm_layer(self, x):
    ''' lstm layers'''
    if self.netconf['use_lstm_layer']:
      with tf.variable_scope('lstm'):
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(
            self.netconf['cell_num'], forget_bias=1.0)
        if self.netconf['use_dropout']:
          cell_fw = tf.nn.rnn_cell.DropoutWrapper(
              cell=cell_fw,
              output_keep_prob=1 -
              self.netconf['dropout_rate'] if self.train else 1.0)

        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(
            self.netconf['cell_num'], forget_bias=1.0)
        if self.netconf['use_dropout']:
          cell_bw = tf.nn.rnn_cell.DropoutWrapper(
              cell=cell_bw,
              output_keep_prob=1 -
              self.netconf['dropout_rate'] if self.train else 1.0)

        # Now we feed `linear` into the LSTM BRNN cell and obtain the LSTM BRNN output.
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=x,
            dtype=tf.float32,
            time_major=False,
            scope='LSTM1')
        del output_states
    else:
      outputs = x
    return outputs

  def pooling_layer(self, x, time_len):
    ''' pooling layer'''
    with tf.variable_scope('time_pooling'):
      if self.attention:
        x, self.alphas = common_layers.attention(
            x, self.netconf['attention_size'], return_alphas=True)
        #alphas shape [batch, time, 1] -> [1, batch, time, 1]-> [1, time, batch, 1]
        tf.summary.image(
            'alignment',
            tf.transpose(tf.expand_dims(self.alphas, 0), [0, 2, 1, 3]))
      else:
        if self.netconf['use_lstm_layer']:
          x = tf.concat(x, 2)
        # [batch, seq_len, dim, 1]
        x = tf.expand_dims(x, axis=-1)
        seq_len = time_len
        x = common_layers.max_pool(x, ksize=[seq_len, 1], strides=[seq_len, 1])
        if self.netconf['use_lstm_layer']:
          x = tf.reshape(x, [-1, 2 * self.netconf['cell_num']])
        else:
          x = tf.reshape(x, [-1, self.netconf['linear_num']])
      return x

  def text_layer(self, x, input_text):
    ''' text embbeding layers'''
    with tf.variable_scope('text'):
      embedding_chars_expanded = common_layers.embedding_look_up(
          input_text, self.vocab_size, self.netconf['embedding_dim'])
      h_pool_flat = common_layers.conv_pool(
          embedding_chars_expanded,
          list(map(int, self.netconf['filter_sizes'])),
          self.netconf['embedding_dim'], self.netconf['num_filters'],
          input_text.shape[1])
      outputs = tf.concat((x, h_pool_flat), axis=1)
    return outputs

  def dense_layer(self, x):
    ''' fc layers'''
    with tf.variable_scope('dense'):
      shape = x.shape[-1].value
      y = common_layers.linear(x, 'dense-matmul',
                               [shape, self.netconf['hidden1']])
      if self.netconf['use_bn']:
        y = tf.layers.batch_normalization(
            y, axis=-1, momentum=0.99, training=self.train, name='dense-bn')
      y = tf.nn.relu6(y)
      if self.netconf['use_dropout']:
        y = tf.layers.dropout(
            y, self.netconf['dropout_rate'], training=self.train)
    return y

  def logits_layer(self, x):
    ''' output layers'''
    with tf.variable_scope('logits'):
      logits = common_layers.linear(
          x, 'logits-matmul',
          [self.netconf['hidden1'], self.taskconf['classes']['num']])
    return logits

  def model(self, inputs, input_text):
    ''' build model '''
    x, downsample_time_len = self.conv_block(inputs, depthwise=False)
    x = self.linear_block(x)
    x = self.lstm_layer(x)
    x = self.pooling_layer(x, downsample_time_len)
    if self.taskconf['text']['enable']:
      x = self.text_layer(x, input_text)
    x = self.dense_layer(x)
    logits = self.logits_layer(x)
    return logits


@registers.model.register
class EmoDCRNNRawModel(EmoCRNNRawModel):
  ''' emotion dcrnn model '''

  def __init__(self):
    super().__init__()
    self.depthwise = True

  def model(self, inputs, input_text):
    x, downsample_time_len = self.conv_block(inputs, depthwise=self.depthwise)
    x = self.linear_block(x)
    x = self.lstm_layer(x)
    x = self.pooling_layer(x, downsample_time_len)
    if self.taskconf['text']['enable']:
      x = self.text_layer(x, input_text)
    x = self.dense_layer(x)
    logits = self.logits_layer(x)
    return logits


@registers.model.register
class EmoNDCRNNRawModel(EmoCRNNRawModel):
  ''' emotion ndcrnn model '''

  def __init__(self):
    super().__init__()
    self.depthwise = True

  def model(self, inputs, input_text):
    x, downsample_time_len = self.conv_block(inputs, depthwise=self.depthwise)
    x = self.linear_block(x)
    x = self.lstm_layer(x)
    x = self.pooling_layer(x, downsample_time_len)
    if self.taskconf['text']['enable']:
      x = self.text_layer(x, input_text)
    x = self.dense_layer(x)
    logits = self.logits_layer(x)
    return logits
