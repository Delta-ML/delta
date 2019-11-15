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
"""Base class for layer."""

import delta.compat as tf


class Layer(tf.keras.layers.Layer):
  """Base class for layer."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def build(self, input_shape):
    """Creates the variables of the layer."""
    #pylint: disable=useless-super-delegation
    super().build(input_shape)

  def call(self, inputs, training=None, mask=None):
    """This is where the layer's logic lives."""
    # pylint: disable=arguments-differ
    raise NotImplementedError()
