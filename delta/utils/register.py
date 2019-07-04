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
"""Module register."""

import importlib
from absl import logging


class Register:
  """Module register"""

  def __init__(self, registry_name):
    self._dict = {}
    self._name = registry_name

  def __setitem__(self, key, value):
    if not callable(value):
      raise Exception("Value of a Registry must be a callable.")
    if key is None:
      key = value.__name__
    if key in self._dict:
      logging.warning("Key %s already in registry %s." % (key, self._name))
    self._dict[key] = value

  def register(self, param):
    """Decorator to register a function or class."""

    def decorator(key, value):
      self[key] = value
      return value

    if callable(param):
      # @reg.register
      return decorator(None, param)
    # @reg.register('alias')
    return lambda x: decorator(param, x)

  def __getitem__(self, key):
    return self._dict[key]

  def __contains__(self, key):
    return key in self._dict

  def keys(self):
    """key"""
    return self._dict.keys()


class registers():  # pylint: disable=invalid-name, too-few-public-methods
  """All module registers."""

  def __init__(self):
    raise RuntimeError("Registries is not intended to be instantiated")

  task = Register('task')
  model = Register('model')
  solver = Register('solver')
  loss = Register('loss')
  metric = Register('metric')
  preparer = Register('preparer')
  preprocessor = Register('preprocessor')
  postprocess = Register('postprocess')
  serving = Register('serving')


module_names = ["delta.data.task",
                "delta.models",
                "delta.utils.loss",
                "delta.utils.metrics",
                "delta.utils.solver",
                "delta.utils.postprocess",
                "delta.serving",
                "delta.data.preprocess"]


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  for name, err in errors:
    logging.warning("Module {} import failed: {}".format(name, err))


def import_all_modules_for_register():
  errors = []
  for name in module_names:
    try:
      importlib.import_module(name)
    except ImportError as error:
      errors.append((name, error))
  _handle_errors(errors)
