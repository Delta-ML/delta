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
"""Base task for NLP."""

from absl import logging

from delta.data.task.base_task import Task
from delta import utils
from delta.utils.register import registers

# pylint: disable=abstract-method


class TextTask(Task):
  """Base task for NLP."""

  def __init__(self, config, mode):
    super().__init__(config)
    self.all_modes = [utils.TRAIN, utils.EVAL, utils.INFER]
    assert mode in self.all_modes
    self.preparer = None
    self.use_preparer = True

  def input_fn(self):

    def _input_fn():
      return self.dataset()

    return _input_fn

  def preprocess_batch(self, batch):
    """
    Pre-process batch.
    This function is not used in all nlp tasks.
    """
    return batch

  def pre_process_pipeline(self, input_sentences):
    """Data pipeline function for pre-processing."""
    raise NotImplementedError

  def common_process_pipeline(self, batch):
    """
    Data pipeline function for common process.
    This function is used both by online training and offline inference.
    """
    raise NotImplementedError

  def get_input_pipeline(self, for_export):
    """Get the input pipeline function."""

    def input_pipeline_func(input_sentences):
      """This input pipeline function will be used in online inference."""
      if for_export:
        input_sentences = self.pre_process_pipeline(input_sentences)
      batch = self.common_process_pipeline(input_sentences)
      return batch

    return input_pipeline_func

  def set_preparer(self):
    """Set the preparer"""
    if 'preparer' not in self.config['data']['task']:
      self.use_preparer = False
    else:
      self.use_preparer = self.config['data']['task']['preparer']["enable"]
    if self.use_preparer:
      preparer_name = self.config['data']['task']['preparer']["name"]
      self.preparer = registers.preparer[preparer_name](self.config)

  def prepare(self):
    """
    Do all steps for pre-processing,
    including putting data to the pre-process pipeline,
    preparing the vocabulary file and preparing the embedding file.
    """
    self.set_preparer()
    if not self.use_preparer:
      logging.info("This task do not has a Preparer.")
      return
    if self.preparer.skip_prepare():
      logging.info("Skip the Preparing process.")
      return
    else:
      self.preparer.do_prepare(self.pre_process_pipeline)
      self.preparer.done_prepare()
