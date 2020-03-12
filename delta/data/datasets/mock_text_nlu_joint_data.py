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


@registers.dataset.register('mock_text_nlu_joint_data')
class MockTextNLUJointData(BaseDataSet):
  """mock nlu-joint data class for nlu-joint task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.dev_file = "dev.txt"
    self.test_file = "test.txt"
    self.data_files = [self.train_file, self.dev_file, self.test_file]
    self.config_files = ['nlu_joint_mock.yml']
    self.download_files = []
    self.text_vocab = "text_vocab.txt"

    # samples with label
    self.samples = ["0\tO O O O\tmy feeling is low",
               "1\tO O O O B-ORG\ti am happy in the kfc"]

    self.text_vocab_list = ["<unk>\t0", "</s>\t1", "i\t2", "am\t3", "kfc\t4", "my\t5",
                       "feeling\t6", "happy\t7", "is\t8", "low\t9", "in\t10", "the\t11"]

  def download(self) -> bool:
    return True


  def after_download(self) -> bool:
    try:
      train_file_path = os.path.join(self.data_dir, self.train_file)
      dev_file_path = os.path.join(self.data_dir, self.dev_file)
      test_file_path = os.path.join(self.data_dir, self.test_file)
      text_vocab_file = os.path.join(self.data_dir, self.text_vocab)

      mock_data(self.samples, train_file_path, dev_file_path,
                test_file_path, text_vocab_file, self.text_vocab_list)

    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
