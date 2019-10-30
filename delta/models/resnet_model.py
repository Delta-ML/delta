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
''' resnet keras model'''
from absl import logging

#pylint: disable=no-name-in-module
import delta.compat as tf
from tensorflow.python.keras import backend as K

from delta.models.base_model import Model
from delta.layers.resnet import IdentityBlock
from delta.layers.resnet import ConvBlock

from delta.utils.register import registers

#pylint: disable=invalid-name
#pylint: disable=attribute-defined-outside-init
#pylint: disable=missing-docstring
#pylint: disable=too-many-instance-attributes
#pylint: disable=attribute-defined-outside-init
#pylint: disable=too-many-ancestors

layers = tf.keras.layers


@registers.model.register
class ResNet50(Model):

  def __init__(self, config, **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    # Reference
        https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
    """
    super().__init__(**kwargs)
    classes = config['data']['task']['classes']['num']
    self.include_top = True
    self.pooling = 'avg'

    if K.image_data_format() == 'channels_last':
      bn_axis = 3
    else:
      bn_axis = 1

    self.zero_pad1 = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')
    self.conv1 = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        kernel_initializer='he_normal',
        name='conv1')
    self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
    self.act1 = layers.Activation('relu')
    self.zero_pad2 = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')
    self.max_pool1 = layers.MaxPooling2D((3, 3), strides=(2, 2))

    self.conv_block1a = ConvBlock(
        3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    self.identity_block1b = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
    self.identity_block1c = IdentityBlock(3, [64, 64, 256], stage=2, block='c')

    self.conv_block2a = ConvBlock(3, [128, 128, 512], stage=3, block='a')
    self.identity_block2b = IdentityBlock(
        3, [128, 128, 512], stage=3, block='b')
    self.identity_block2c = IdentityBlock(
        3, [128, 128, 512], stage=3, block='c')
    self.identity_block2d = IdentityBlock(
        3, [128, 128, 512], stage=3, block='d')

    self.conv_block3a = ConvBlock(3, [256, 256, 1024], stage=4, block='a')
    self.identity_block3b = IdentityBlock(
        3, [256, 256, 1024], stage=4, block='b')
    self.identity_block3c = IdentityBlock(
        3, [256, 256, 1024], stage=4, block='c')
    self.identity_block3d = IdentityBlock(
        3, [256, 256, 1024], stage=4, block='d')
    self.identity_block3e = IdentityBlock(
        3, [256, 256, 1024], stage=4, block='e')
    self.identity_block3f = IdentityBlock(
        3, [256, 256, 1024], stage=4, block='f')

    self.conv_block4a = ConvBlock(3, [512, 512, 2048], stage=5, block='a')
    self.identity_block4b = IdentityBlock(
        3, [512, 512, 2048], stage=5, block='b')
    self.identity_block4c = IdentityBlock(
        3, [512, 512, 2048], stage=5, block='c')

    self.global_avg_pool = layers.GlobalAveragePooling2D(name='avg_pool')
    self.dense = layers.Dense(classes, activation='softmax', name='fc-class')

  #pylint: disable=arguments-differ
  def call(self, input_tensor, training=None, mask=None):
    logging.info(f"input: {input_tensor}")
    if isinstance(input_tensor, (tuple, list)):
      x = input_tensor[0]
    elif isinstance(input_tensor, dict):
      x = input_tensor['inputs']
    else:
      x = input_tensor

    x = self.zero_pad1(x)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.zero_pad2(x)
    x = self.max_pool1(x)

    x = self.conv_block1a(x)
    x = self.identity_block1b(x)
    x = self.identity_block1c(x)

    x = self.conv_block2a(x)
    x = self.identity_block2b(x)
    x = self.identity_block2c(x)
    x = self.identity_block2d(x)

    x = self.conv_block3a(x)
    x = self.identity_block3b(x)
    x = self.identity_block3c(x)
    x = self.identity_block3d(x)
    x = self.identity_block3e(x)
    x = self.identity_block3f(x)

    x = self.conv_block4a(x)
    x = self.identity_block4b(x)
    x = self.identity_block4c(x)

    x = self.global_avg_pool(x)
    x = self.dense(x)
    return x
