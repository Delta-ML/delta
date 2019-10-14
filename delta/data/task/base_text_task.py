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

import os
from absl import logging

from delta.data.task.base_task import Task
from delta import utils
from delta.utils.register import registers
from delta.data.preprocess.text_ops import clean_english_str_tf
from delta.data.preprocess.text_ops import char_cut_tf
from delta.data.preprocess.text_ops import tokenize_sentence
from delta.data.preprocess.text_ops import chinese_word_cut_tf

# pylint: disable=abstract-method


class TextTask(Task):
  """Base task for NLP."""

  def __init__(self, config, mode):
    super().__init__(config)
    self.all_modes = [utils.TRAIN, utils.EVAL, utils.INFER]
    assert mode in self.all_modes
    self.preparer = None
    self.use_preparer = True
    self.mode = mode
    self.model_config = config["model"]
    self.data_config = config['data']
    self.task_config = self.data_config['task']

    self.infer_no_label = self.data_config[utils.INFER].get(
        'infer_no_label', False)
    if self.mode == utils.INFER and self.infer_no_label:
      self.infer_without_label = True
    else:
      self.infer_without_label = False

    self.batch_size = self.task_config['batch_size']
    self.epochs = self.task_config['epochs']
    self.num_parallel_calls = self.task_config['num_parallel_calls']
    self.num_prefetch_batch = self.task_config['num_prefetch_batch']
    self.shuffle_buffer_size = self.task_config['shuffle_buffer_size']
    self.need_shuffle = self.task_config['need_shuffle']


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
    language = self.task_config["language"]
    clean_english = self.task_config.get("clean_english", False)
    split_by_space = self.task_config.get("split_by_space", False)
    use_word = self.task_config.get("use_word", False)

    if language == "english":
      if clean_english:
        batch = clean_english_str_tf(input_sentences)
      else:
        batch = input_sentences
    else:
      if split_by_space:
        batch = input_sentences
      else:
        if use_word:
          batch = chinese_word_cut_tf(input_sentences)
        else:
          batch = char_cut_tf(input_sentences)
    return batch

  def common_process_pipeline(self, batch):
    """
    Data pipeline function for common process.
    This function is used both by online training and offline inference.
    """
    text_vocab_file_path = self.task_config['text_vocab']
    max_seq_len = self.task_config['max_seq_len']
    vocab_path = os.path.abspath(text_vocab_file_path)
    token_ids = tokenize_sentence(batch, max_seq_len, vocab_path)
    return token_ids

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

    self.preparer.do_prepare(self.pre_process_pipeline)
    self.preparer.done_prepare()
