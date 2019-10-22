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
"""Common layers test."""

import delta.compat as tf
from absl import logging

import common_layers as cl


class LossUtilTest(tf.test.TestCase):
  ''' common layer unittest '''

  def test_splice_layer(self):
    '''test splice layer'''
    inputs = tf.reshape(tf.range(15), shape=[1, 5, 3])
    context = [0, 1]
    output = cl.splice_layer(inputs, 'splice', context)
    output_true = tf.constant([[[0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8],
                                [6, 7, 8, 9, 10, 11], [9, 10, 11, 12, 13, 14],
                                [12, 13, 14, 12, 13, 14]]])
    self.assertAllEqual(output, output_true)

    context = [-1, 0, 1]
    output = cl.splice_layer(inputs, 'splice', context)
    output_true = tf.constant([[[0, 1, 2, 0, 1, 2, 3, 4, 5],
                                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                [3, 4, 5, 6, 7, 8, 9, 10, 11],
                                [6, 7, 8, 9, 10, 11, 12, 13, 14],
                                [9, 10, 11, 12, 13, 14, 12, 13, 14]]])
    self.assertAllEqual(output, output_true)

    context = [0, 1, 3]
    output = cl.splice_layer(inputs, 'splice', context)
    output_true = tf.constant([[[0, 1, 2, 3, 4, 5, 9, 10, 11],
                                [3, 4, 5, 6, 7, 8, 12, 13, 14],
                                [6, 7, 8, 9, 10, 11, 12, 13, 14],
                                [9, 10, 11, 12, 13, 14, 12, 13, 14],
                                [12, 13, 14, 12, 13, 14, 12, 13, 14]]])
    self.assertAllEqual(output, output_true)

    context = [1, 3]
    output = cl.splice_layer(inputs, 'splice', context)
    output_true = tf.constant([[[3, 4, 5, 9, 10, 11], [6, 7, 8, 12, 13, 14],
                                [9, 10, 11, 12, 13, 14],
                                [12, 13, 14, 12, 13, 14],
                                [12, 13, 14, 12, 13, 14]]])
    self.assertAllEqual(output, output_true)

    context = [1, 2, 3]
    output = cl.splice_layer(inputs, 'splice', context)
    output_true = tf.constant([[[3, 4, 5, 6, 7, 8, 9, 10, 11],
                                [6, 7, 8, 9, 10, 11, 12, 13, 14],
                                [9, 10, 11, 12, 13, 14, 12, 13, 14],
                                [12, 13, 14, 12, 13, 14, 12, 13, 14],
                                [12, 13, 14, 12, 13, 14, 12, 13, 14]]])
    self.assertAllEqual(output, output_true)

  def test_tdnn(self):
    '''test tdnn'''
    #A 3D Tensor [batch, in_width, in_channels]
    inputs = tf.random_uniform(shape=[2, 5, 3], dtype=tf.float32, maxval=1.0)
    in_dim = inputs.get_shape().as_list()[2]
    out_dim = 4
    context = [-2, -1, 0, 1, 2]
    output = cl.tdnn(
        inputs, 'test_tdnn0', in_dim, context, out_dim, method='splice_layer')
    out_shape = [2, 5, 4]
    self.assertAllEqual(tf.shape(output), out_shape)

    context = 2
    #output = cl.tdnn(inputs, 'test_tdnn1', in_dim, context, out_dim, method='splice_op')
    #self.assertAllEqual(tf.shape(output), out_shape)

    output = cl.tdnn(
        inputs, 'test_tdnn2', in_dim, context, out_dim, method='conv1d')
    self.assertAllEqual(tf.shape(output), out_shape)

  def test_conv2d(self):
    '''test conv2d'''
    inputs = tf.random_uniform(
        shape=[2, 5, 5, 3], dtype=tf.float32, maxval=1.0)  #A 4D Tensor
    filter_size = [3, 3]
    in_channels = inputs.get_shape().as_list()[3]
    out_channels = 4
    strides = [1, 1]
    output = cl.conv2d(inputs, 'test_conv2d', filter_size, in_channels,
                       out_channels, strides)
    output_shape = [2, 5, 5, 4]
    self.assertAllEqual(tf.shape(output), output_shape)

  def test_maxpool(self):
    '''test maxpool'''
    inputs = tf.reshape(tf.range(25), shape=[1, 5, 5, 1])  #A 4D tensor
    ksize = [3, 3]
    strides = [1, 1]
    output = cl.max_pool(inputs, ksize, strides)
    output_shape = [1, 3, 3, 1]
    self.assertAllEqual(tf.shape(output), output_shape)

    output_true = tf.constant([[[[12], [13], [14]], [[17], [18], [19]],
                                [[22], [23], [24]]]])
    self.assertAllEqual(output, output_true)

  def test_linear(self):
    '''test linear'''
    inputs = tf.random_uniform(
        shape=[4, 5], dtype=tf.float32, maxval=1.0)  # A 2D tensor
    shape = [5, 4]
    output = cl.linear(inputs, 'test_linear0', shape)
    output_shape = [4, 4]
    self.assertAllEqual(tf.shape(output), output_shape)

    inputs = tf.random_uniform(
        shape=[2, 4, 5], dtype=tf.float32, maxval=1.0)  # A 3D tensor
    shape = [5, 4]
    output = cl.linear(inputs, 'test_linear1', shape)
    output_shape = [2, 4, 4]
    self.assertAllEqual(tf.shape(output), output_shape)

    # A 4D tensor [B, C, H, W]
    inputs = tf.random_uniform(shape=[2, 3, 4, 5], dtype=tf.float32, maxval=1.0)
    shape = [5, 4]
    output = cl.linear(inputs, 'test_linear2', shape)
    output_shape = [2, 3, 4, 4]
    self.assertAllEqual(tf.shape(output), output_shape)

  def test_attention(self):
    '''test attention'''
    # A 3D tensor [B, T, D]
    inputs = tf.random_uniform(
        shape=[2, 100, 512], dtype=tf.float32, maxval=1.0)
    attention_size = 256
    output, alpha = cl.attention(inputs, attention_size, return_alphas=True)
    output_shape = [2, 512]
    alpha_shape = [2, 100, 1]
    self.assertAllEqual(tf.shape(output), output_shape)
    self.assertAllEqual(tf.shape(alpha), alpha_shape)

  def test_embedding_look_up(self):
    '''test embedding look up'''
    text_inputs = [0, 1, 2]
    vocab_size = 3
    embedding_size = 512
    output = cl.embedding_look_up(text_inputs, vocab_size, embedding_size)
    output_shape = [3, 512, 1]
    self.assertAllEqual(tf.shape(output), output_shape)

  def test_conv_pool(self):
    '''test  conv pool'''
    # A 4D tensor [B, H, W, C]
    embedded_chars_expanded = tf.random_uniform(
        shape=[2, 7, 7, 1], dtype=tf.float32, maxval=1.0)
    filter_sizes = [3, 5]
    embedding_size = 3
    num_filters = 3
    sequence_length = 5
    output = cl.conv_pool(embedded_chars_expanded, filter_sizes, embedding_size,
                          num_filters, sequence_length)
    output_shape = [30, 6]
    self.assertAllEqual(tf.shape(output), output_shape)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
