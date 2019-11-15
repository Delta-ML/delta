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
''' resnet layers'''
#pylint: disable=no-name-in-module
import delta.compat as tf
from tensorflow.python.keras import backend as K

from delta.layers.base_layer import Layer

#pylint: disable=invalid-name
#pylint: disable=attribute-defined-outside-init
#pylint: disable=missing-docstring
#pylint: disable=too-many-instance-attributes
#pylint: disable=attribute-defined-outside-init
#pylint: disable=too-many-ancestors

layers = tf.keras.layers


class IdentityBlock(Layer):

  def __init__(self, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
      # Arguments
          kernel_size: default 3, the kernel size of
              middle conv layer at main path
          filters: list of integers, the filters of 3 conv layer at main path
          stage: integer, current stage label, used for generating layer names
          block: 'a','b'..., current block label, used for generating layer names
      # Returns
          Output tensor for the block.
    """
    super().__init__(name='identity' + str(stage) + block)
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.conv1 = layers.Conv2D(
        filters1, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2a')
    self.bn1 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
    self.act1 = layers.Activation('relu')

    self.conv2 = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        name=conv_name_base + '2b')
    self.bn2 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
    self.act2 = layers.Activation('relu')

    self.conv3 = layers.Conv2D(
        filters3, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2c')
    self.bn3 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

    self.add = layers.Add()
    self.act = layers.Activation('relu')

  #pylint: disable=arguments-differ
  def call(self, input_tensor, training=None, mask=None):
    x = self.conv1(input_tensor)
    x = self.bn1(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.act2(x)

    x = self.conv3(x)
    x = self.bn3(x)

    x = self.add([x, input_tensor])
    x = self.act(x)
    return x


class ConvBlock(Layer):

  #pylint: disable=too-many-arguments
  def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    super().__init__(name='conv_block' + str(stage) + block)
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    self.conv1 = layers.Conv2D(
        filters1, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '2a')
    self.bn1 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
    self.act1 = layers.Activation('relu')

    self.conv2 = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        name=conv_name_base + '2b')
    self.bn2 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
    self.act2 = layers.Activation('relu')

    self.conv3 = layers.Conv2D(
        filters3, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2c')
    self.bn3 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

    self.shortcut_conv = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')
    self.shortcut_bn = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')

    self.add = layers.Add()
    self.act = layers.Activation('relu')

  #pylint: disable=arguments-differ
  def call(self, input_tensor, training=None, mask=None):
    x = self.conv1(input_tensor)
    x = self.bn1(x)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.act2(x)

    x = self.conv3(x)
    x = self.bn3(x)

    shortcut = self.shortcut_conv(input_tensor)
    shortcut = self.shortcut_bn(shortcut)

    x = self.add([x, shortcut])
    x = self.act(x)
    return x
