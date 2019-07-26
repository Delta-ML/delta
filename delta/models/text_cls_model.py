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
"""Base class for text classification."""

import tensorflow as tf

from delta.models.base_model import Model

# pylint: disable=abstract-method,too-many-ancestors
class TextClassModel(Model):
  """Base class for text classification."""

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    model_config = config['model']['net']['structure']
    if "dense_input" in model_config:
      self.use_dense_task = True
      self.use_dense_input = model_config["dense_input"]["use_dense_input"]
      self.only_dense_input = model_config["dense_input"]["only_dense_input"]
      self.dense_input_dim = config['data']['task']['dense_input_dim']
    else:
      self.use_dense_task = False
      self.use_dense_input = False

    if self.use_dense_input:
      self.dense_input_linear = tf.keras.layers.Dense(
          self.dense_input_dim, activation=tf.keras.activations.linear)
