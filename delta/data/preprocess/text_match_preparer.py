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
''' Preparer for text match.'''

from delta.data.preprocess.base_preparer import TextPreparer
from delta.data import utils as data_utils
from delta.utils.register import registers
from delta.data.preprocess.text_ops import load_raw_data
# pylint: disable=too-many-instance-attributes


@registers.preparer.register
class TextMatchPreparer(TextPreparer):
  """Preparer for text match."""

  def __init__(self, config):
    super().__init__(config)
    self.multi_text = True

  def load_a_raw_file(self, one_path, mode, infer_without_label):
    """
    Load a raw file. Return text and label.
    For single text input, text: [sentence1, ...]
    For multiple text inputs, text: [[sentence1_1, ...], [sentence1_2, ...]]
    For single output, label: [label1, label2, ...]
    For multiple outputs, label: [[label1_1, ...], [label1_2, ...]]
    """

    if infer_without_label:
      col=2
    else:
      col=3

    map_text = load_raw_data([one_path], col)
    if infer_without_label and len(map_text)==2:
      text = map_text
      label = []
    else:
      text = map_text[1:]
      label = map_text[0]

    return (text,label)

  def save_a_raw_file(self, label, text_after, one_path_after,
                      infer_without_label):
    """Save a raw file."""
    data_utils.save_a_text_match_file(label, text_after, one_path_after,
                                      infer_without_label)
