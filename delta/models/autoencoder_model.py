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
''' AutoEncoder keras model'''
import numpy as np
from absl import logging

#pylint: disable=no-name-in-module
import tensorflow as tf

from tensorflow.python.keras import backend as K
from delta.models.base_model import Model

from delta import utils
from delta.utils.hparam import HParams
from delta.utils.register import registers

#pylint: disable=invalid-name
#pylint: disable=attribute-defined-outside-init
#pylint: disable=missing-docstring
#pylint: disable=too-many-instance-attributes
#pylint: disable=attribute-defined-outside-init
#pylint: disable=too-many-ancestors

@registers.model.register
class AEModel(Model):
    """a basic autoencoder class for tensorflow
    Extends:
        tf.keras.Model
    """
    @classmethod
    def params(cls, config: dict=None):
      z_dim = 64
      hp = HParams(cls=cls, name=cls.__name__)
      hp.add_hparam('z_dim', z_dim)
      hp.add_hparam('dims', (28, 28, 1))
      return hp

    def __init__(self, config: dict, **kwargs):
        super().__init__()
        self.config = config
        logging.info(f"config: {self.config}")
        logging.info(f"z dim: {self.config.z_dim}")
        logging.info(f"kwargs {kwargs}")

        self.enc = [
            tf.keras.layers.InputLayer(input_shape=self.config.dims),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.config.z_dim),
        ]
        self.dec = [
             tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
             tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
             tf.keras.layers.Conv2DTranspose(
                 filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
             ),
             tf.keras.layers.Conv2DTranspose(
                 filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
             ),
             tf.keras.layers.Conv2DTranspose(
                 filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
             ),
        ]

        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)

        # as last to override self.enc or self.dec, etc. 
        self.__dict__.update(kwargs)

    @tf.function
    def call(self, inputs, training=None, mask=None):
      ''' just for inference '''
      return self.encode(inputs)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)
    
    def compute_loss(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        return ae_loss
    
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)

    @tf.function
    def train(self, train_x):    
        gradients = self.compute_gradients(train_x)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
