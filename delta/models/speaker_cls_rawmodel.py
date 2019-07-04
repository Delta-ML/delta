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
'''
A simple CRNN model for speaker classification.
Currently serves as a proof-of-concept, needs overhaul.
'''
from absl import logging
import tensorflow as tf

from delta import utils
from delta.layers import common_layers
from delta.models.base_model import RawModel
from delta.utils.register import registers

#pylint: disable=invalid-name
#pylint: disable=too-many-locals
#pylint: disable=too-many-instance-attributes
#pylint: disable=arguments-differ


@registers.model.register
class SpeakerCRNNRawModel(RawModel):
  ''' A simple speaker model. '''

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

    # internal parameters
    self.feature_params = None
    self.mean = None
    self.std = None
    self.train = None

  def preprocess(self, inputs, input_text):
    ''' Speech preprocessing. '''
    with tf.variable_scope('feature'):
      if self.input_type == 'samples':
        # FIXME: stub
        feats = None
      else:
        if 'cmvn_type' in self.audioconf:
          cmvn_type = self.audioconf['cmvn_type']
        else:
          cmvn_type = 'global'
        logging.info('cmvn_type: %s' % (cmvn_type))
        if cmvn_type == 'global':
          self.mean, self.std = utils.load_cmvn(self.audioconf['cmvn_path'])
          feats = utils.apply_cmvn(inputs, self.mean, self.std)
        elif cmvn_type == 'local':
          feats = utils.apply_local_cmvn(inputs)
        elif cmvn_type == 'sliding':
          raise ValueError('cmvn_type %s not implemented yet.' % (cmvn_type))
        else:
          raise ValueError('Error cmvn_type %s.' % (cmvn_type))
    return feats, input_text

  def call(self, features, **kwargs):
    ''' Implementation of __call__(). '''
    self.train = kwargs['training']
    feats = tf.identity(features['inputs'], name='feats')
    texts = features['texts']

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      feats, texts = self.preprocess(feats, texts)
      logits = self.model(feats, texts)
    return logits

  def conv_block(self, inputs, depthwise=False):
    ''' 2D conv layers. '''
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
      for index, filt in enumerate(filters):
        unit_name = 'unit-' + str(index + 1)
        with tf.variable_scope(unit_name):
          if depthwise:
            x = tf.layers.separable_conv2d(
                x,
                filters=filt,
                kernel_size=filters_size[index],
                strides=filters_strides[index],
                padding='same',
                name=unit_name)
          else:
            cnn_name = 'cnn-' + str(index + 1)
            x = common_layers.conv2d(x, cnn_name, filters_size[index],
                                     channels[index], channels[index + 1],
                                     filters_strides[index])
          x = tf.nn.relu(x)
          if self.netconf['use_bn']:
            bn_name = 'bn' + str(index + 1)
            x = tf.layers.batch_normalization(
                x, axis=-1, momentum=0.9, training=self.train, name=bn_name)
          if self.netconf['use_dropout']:
            x = tf.layers.dropout(
                x, self.netconf['dropout_rate'], training=self.train)
          x = common_layers.max_pool(x, pools_size[index], pools_size[index])
          downsample_input_len = downsample_input_len / pools_size[index][0]

    return x, downsample_input_len

  def tdnn_block(self, inputs):
    ''' TDNN layers. '''
    tdnn_contexts = self.netconf['tdnn_contexts']
    logging.info("tdnn_contexts : {}".format(tdnn_contexts))
    tdnn_dims = self.netconf['tdnn_dims']
    logging.info("tdnn_dims : {}".format(tdnn_dims))

    layer_num = len(tdnn_contexts)
    assert layer_num == len(tdnn_dims)

    channels = [self.input_channels] + tdnn_dims
    logging.info("tdnn_channels : {}".format(channels))

    # NHWC -> NW'C, W' = H * W
    input_n, input_h, input_w, input_c = inputs.shape.as_list()
    inputs = tf.reshape(inputs, [-1, input_h * input_w, input_c])

    downsample_input_len = self.input_len
    with tf.variable_scope('tdnn'):
      x = tf.identity(inputs)
      for index in range(layer_num):
        unit_name = 'unit-' + str(index + 1)
        with tf.variable_scope(unit_name):
          tdnn_name = 'tdnn-' + str(index + 1)
          use_bn = self.netconf['use_bn']
          has_bias = not use_bn
          x = common_layers.tdnn(
              x,
              tdnn_name,
              channels[index],
              tdnn_contexts[index],
              channels[index + 1],
              has_bias=has_bias)
          x = tf.nn.relu(x)
          if self.netconf['use_bn']:
            bn_name = 'bn' + str(index + 1)
            x = tf.layers.batch_normalization(
                x, axis=-1, momentum=0.9, training=self.train, name=bn_name)
          if self.netconf['use_dropout']:
            x = tf.layers.dropout(
                x, self.netconf['dropout_rate'], training=self.train)
          downsample_input_len = downsample_input_len

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
      x = tf.nn.relu(x)
      if self.netconf['use_bn']:
        bn_name = 'bn_linear'
        x = tf.layers.batch_normalization(
            x, axis=-1, momentum=0.9, training=self.train, name=bn_name)
      x = tf.reshape(x, [-1, times, self.netconf['linear_num']])
    return x

  def lstm_layer(self, x):
    ''' LSTM layers. '''
    if self.netconf['use_lstm_layer']:
      with tf.variable_scope('lstm'):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(
            self.netconf['cell_num'], forget_bias=1.0)
        if self.netconf['use_dropout']:
          cell_fw = tf.contrib.rnn.DropoutWrapper(
              cell=cell_fw,
              output_keep_prob=1 -
              self.netconf['dropout_rate'] if self.train else 1.0)

        cell_bw = tf.contrib.rnn.BasicLSTMCell(
            self.netconf['cell_num'], forget_bias=1.0)
        if self.netconf['use_dropout']:
          cell_bw = tf.contrib.rnn.DropoutWrapper(
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
    else:
      outputs = x
    return outputs

  def stats_pooling_layer(self, x):
    '''
      Statistics pooling layer.
      Input: [NHW]
        --> Reduce along H
      Output: [NW'] where W' = W * 2
    '''
    with tf.variable_scope('stats_pooling'):
      mean, var = tf.nn.moments(x, 1)
      x = tf.concat([mean, tf.sqrt(var + 1e-6)], 1)
    return x

  def text_layer(self, x, input_text):
    ''' Text layer. Might be useless in speaker model. '''
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
    ''' Embedding layers. '''
    with tf.variable_scope('dense'):
      shape = x.shape[-1].value
      if 'hidden_dims' in self.netconf:
        hidden_dims = self.netconf['hidden_dims']
      else:
        hidden_dims = [self.netconf['hidden1']]
      hidden_idx = 1
      y = x
      for hidden in hidden_dims:
        y = common_layers.linear(y, 'dense-matmul-%d' % (hidden_idx),
                                 [shape, hidden])
        shape = hidden
        y = tf.nn.relu(y)
        if self.netconf['use_bn']:
          y = tf.layers.batch_normalization(
              y,
              axis=-1,
              momentum=0.99,
              training=self.train,
              name='dense-bn-%d' % (hidden_idx))
        if self.netconf['use_dropout']:
          y = tf.layers.dropout(
              y, self.netconf['dropout_rate'], training=self.train)
        hidden_idx += 1
    return y

  def logits_layer(self, x):
    ''' Logits layer to further produce softmax. '''
    with tf.variable_scope('logits'):
      logits = common_layers.linear(
          x, 'logits-matmul',
          [x.shape[-1].value, self.taskconf['classes']['num']])
    return logits

  def model(self, inputs, input_text):
    ''' Build the model. '''
    if 'tdnn_contexts' in self.netconf:
      x, _ = self.tdnn_block(inputs)
    else:
      x, _ = self.conv_block(inputs, depthwise=False)
      x = self.linear_block(x)
      x = self.lstm_layer(x)
    x = self.stats_pooling_layer(x)
    if self.taskconf['text']['enable']:
      x = self.text_layer(x, input_text)
    dense_output = self.dense_layer(x)
    logits = self.logits_layer(dense_output)
    model_outputs = {'logits': logits, 'embeddings': dense_output}
    return model_outputs
