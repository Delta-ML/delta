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

"""
## Data scale introduction


"""

import os
import traceback
from absl import logging
from delta.data.datasets.base_dataset import BaseDataSet
from delta.data.datasets.utils import mock_data
from delta.utils.register import registers


@registers.dataset.register('mock_text_seq2seq_data')
class MockTextSeq2SeqData(BaseDataSet):
  """mock seq2seq data class seq2seq task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.dev_file = "dev.txt"
    self.test_file = "test.txt"

    files = [self.train_file, self.dev_file, self.test_file]
    suffix = [".tgt", ".src"]
    self.data_files = [file+suf for suf in suffix for file in files]

    self.config_files = ['transformer_s2s_mock.yml']
    self.download_files = []


    # samples with label
    self.samples = [" a shooting at a bar popular with expatriates in mali on saturday killed "
               "five people \t killed five people",
               "a pennsylvania community is pulling together to search for an eighth-grade "
               "student who has been missing since wednesday\tsearch for missing student"]

  def download(self) -> bool:
    return True


  def after_download(self) -> bool:
    try:
      train_file_path = os.path.join(self.data_dir, self.train_file)
      dev_file_path = os.path.join(self.data_dir, self.dev_file)
      test_file_path = os.path.join(self.data_dir, self.test_file)

      mock_data(self.samples, train_file_path, dev_file_path, test_file_path)

    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
