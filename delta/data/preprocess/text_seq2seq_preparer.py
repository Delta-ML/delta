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
''' Preparer for text sequence to sequence.'''

import os
from delta import utils
from absl import logging
from delta.data.preprocess.base_preparer import TextPreparer
from delta.data import utils as data_utils
from delta.utils.register import registers
from delta.data.preprocess.utils import prepare_vocab
from delta.data.preprocess.utils import prepare_vocab_from_config
from delta.data.preprocess.text_ops import load_textline_dataset
from delta.data.utils.common_utils import get_file_len
# pylint: disable=too-many-instance-attributes, too-many-locals


@registers.preparer.register
class TextS2SPreparer(TextPreparer):
  """Preparer for sequence labeling."""

  def __init__(self, config):
    super().__init__(config)
    self.multi_text = True

  def prepare_raw_data(self, pre_process_pipeline):
    """
    Preparing raw data.
    For all kinds of text input, all_texts: [sentence1, ...]
    For single output, all_labels: [label1, label2, ...]
    For multiple outputs, all_labels: [[label1_1, ...], [label1_2, ...]]
    """
    if self.output_num <= 1:
      all_labels = []
    else:
      all_labels = [[] for _ in range(self.output_num)]
    all_texts = []
    for mode in self.all_modes:
      paths = self.config["data"][mode]['paths']
      paths = [paths['source'], paths['target']]
      paths_after_pre_process = [
          [one_path + ".after" for one_path in path] for path in paths
      ]
      logging.debug(
          "paths_after_pre_process: {}".format(paths_after_pre_process))

      infer_without_label = bool(mode == utils.INFER and self.infer_no_label)

      for one_path_text, one_path_target, \
          one_path_text_after, one_path_target_after in zip(*paths, *paths_after_pre_process):
        data_size = get_file_len([one_path_text])
        self.prepare_one_raw_data((one_path_text, one_path_target),
                                  (one_path_text_after, one_path_target_after),
                                  mode, infer_without_label,
                                  pre_process_pipeline, all_texts, all_labels,data_size)
    return all_texts, all_labels

  def load_a_raw_file(self, one_path, infer_without_label):
    """
    Load a raw file. Return text and label.
    For single text input, text: [sentence1, ...]
    For multiple text inputs, text: [[sentence1_1, ...], [sentence1_2, ...]]
    For single output, label: [label1, label2, ...]
    For multiple outputs, label: [[label1_1, ...], [label1_2, ...]]
    """
    column_num = 1
    text_path, target_path = one_path
    texts = load_textline_dataset([text_path], column_num)
   # texts = data_utils.load_seq2seq_raw_data([text_path])
    if not infer_without_label:
      target = load_textline_dataset([target_path],column_num)
      return texts+target, target
    return texts, []

  def save_a_raw_file(self, label, text_after, one_path_after,
                      infer_without_label):
    text_path, target_path = one_path_after
    if infer_without_label:
      text = text_after[0]
    else:
      text, target = text_after
      data_utils.save_a_text_seq2seq_file(target, target_path)
    data_utils.save_a_text_seq2seq_file(text, text_path)

  def prepare_label_vocab(self, all_labels):
    """Prepare label vocab"""
    for i in range(self.output_num):
      if os.path.exists(self.label_vocab_file_paths[i]) and \
        self.use_custom_vocab:
        logging.info("Reuse label vocab file: {}".format(
            self.label_vocab_file_paths[i]))
      else:
        prepare_vocab(
            self.label_vocab_file_paths[i],
            all_labels[i],
            min_frequency=self.vocab_min_frequency,
            use_default_dict=True)
        logging.info("Generate label vocab file: {}".format(
            self.label_vocab_file_paths[i]))
