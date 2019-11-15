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
"""Base class for model."""

import re
import delta.compat as tf
from tensorflow.python.keras import backend as K  # pylint: disable=no-name-in-module


class Model(tf.keras.Model):
  """Base class for model."""

  def __init__(self, **kwargs):  # pylint: disable=useless-super-delegation
    super().__init__(**kwargs)

  def __setattr__(self, key, value):
    if key.startswith("temp_"):
      # this is for temporary attributes avoiding keras check
      self.__dict__[key] = value
    else:
      super().__setattr__(key, value)

  def call(self, inputs, training=None, mask=None):
    raise NotImplementedError()


class RawModel:
  """Raw model."""

  def __init__(self, **kwargs):
    name = kwargs.get('name')
    if not name:
      prefix = self.__class__.__name__
      name = self._to_snake_case(prefix) + '_' + str(K.get_uid(prefix))
    self.name = name

  @staticmethod
  def _to_snake_case(name):
    """Transform name to snake case."""
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
      return insecure
    return 'private' + insecure

  def __call__(self, inputs, **kwargs):
    with tf.variable_scope(self.name):
      return self.call(inputs, **kwargs)

  def call(self, inputs, **kwargs):
    """call"""
    raise NotImplementedError()
