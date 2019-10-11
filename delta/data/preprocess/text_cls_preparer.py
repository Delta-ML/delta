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
''' Preparer for text classification.'''

from delta.data.preprocess.base_preparer import TextPreparer
from delta.data.preprocess.text_ops import load_textline_dataset
from delta.data import utils as data_utils
from delta.utils.register import registers


# pylint: disable=too-many-instance-attributes


@registers.preparer.register
class TextClsPreparer(TextPreparer):
  """Preparer for text classification."""

  def __init__(self, config):
    super().__init__(config)

  def load_a_raw_file(self, one_path, infer_without_label):
    """
    Load a raw file. Return text and label.
    For single text input, text: [sentence1, ...]
    For multiple text inputs, text: [[sentence1_1, ...], [sentence1_2, ...]]
    For single output, label: [label1, label2, ...]
    For multiple outputs, label: [[label1_1, ...], [label1_2, ...]]
    """
    if infer_without_label:
      column_num = 1
    else:
      column_num = 2
    ds_list = load_textline_dataset(one_path, column_num)
    if infer_without_label:
      text = ds_list
      label = []         #to modifiy
    else:
      text = ds_list[1:]
      label = ds_list[:1]
    return (text, label)


  def save_a_raw_file(self, label, text_after, one_path_after,
                      infer_without_label):
    """Save a raw file."""
    data_utils.save_a_text_cls_file(label, text_after, one_path_after,
                                    infer_without_label)
