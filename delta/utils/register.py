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
import os
import sys
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


TASK_MODULES = [
    "asr_seq_task", "kws_cls_task", "speech_cls_task", "speech_cls_task",
    "text_cls_task", "text_seq_label_task", "text_match_task",
    "text_nlu_joint_task", "speaker_cls_task", "text_seq2seq_task"
]

MODLE_MODULES = [
    "speech_cls_rawmodel",
    "speaker_cls_rawmodel",
    "speech_cls_model",
    "kws_model",
    "asr_model",
    "text_seq_model",
    "text_hierarchical_model",
    "text_seq_label_model",
    "resnet_model",
    "text_nlu_joint_model",
    "text_match_model",
    "text_seq_label_model",
    "text_seq2seq_model",
]

LOSS_MODULES = ["loss_impl"]

METRICS_MODULES = ["py_metrics"]

SOLVER_MODULES = [
    "raw_cls_solver",
    "raw_match_solver",
    "keras_solver",
    "emotion_solver",
    "kws_solver",
    "asr_solver",
    "speaker_solver",
    "raw_seq_label_solver",
    "raw_nlu_joint_solver",
    "raw_seq2seq_solver",
    "raw_pretrain_cls_solver",
    "raw_pretrain_seq_label_solver",
]

POSTPROCESS_MODULES = [
    "speech_cls_proc", "speaker_cls_proc", "text_cls_proc",
    "text_seq_label_proc", "text_seq2seq_proc"
]

SERVING_MODULES = ["knowledge_distilling"]

PREPROCESS_MODULES = [
    "text_cls_preparer", "text_match_preparer", "text_seq_label_preparer",
    "text_nlu_joint_preparer", "text_seq2seq_preparer"
]

ALL_SYS_MODULES = [("delta.data.task", TASK_MODULES),
                   ("delta.models", MODLE_MODULES),
                   ("delta.utils.loss", LOSS_MODULES),
                   ("delta.utils.metrics", METRICS_MODULES),
                   ("delta.utils.solver", SOLVER_MODULES),
                   ("delta.utils.postprocess", POSTPROCESS_MODULES),
                   ("delta.serving", SERVING_MODULES),
                   ("delta.data.preprocess", PREPROCESS_MODULES)]


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  for name, err in errors:
    logging.warning("Module {} import failed: {}".format(name, err))


def path_to_module_format(py_path):
  """Transform a python file path to module format."""
  return py_path.replace("/", ".").rstrip(".py")


def import_all_modules_for_register(config=None):
  """Import all modules for register."""
  logging.warning("test")
  current_word_dir = os.getcwd()
  if current_word_dir not in sys.path:
    sys.path.append(current_word_dir)
  all_modules = ALL_SYS_MODULES
  errors = []
  if config is not None and "custom_modules" in config:
    custom_modules = config["custom_modules"]
    if not isinstance(custom_modules, list):
      custom_modules = [custom_modules]
    all_modules += [("", [path_to_module_format(module)])
                    for module in custom_modules]
  logging.debug(f"All modules: {all_modules}")
  for base_dir, modules in all_modules:
    for name in modules:
      try:
        if base_dir != "":
          full_name = base_dir + "." + name
        else:
          full_name = name
        importlib.import_module(full_name)
        logging.debug(f"{full_name} loaded.")
      except ImportError as error:
        errors.append((name, error))
  _handle_errors(errors)
