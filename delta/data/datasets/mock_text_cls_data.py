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


@registers.dataset.register('mock_text_cls_data')
class MockTextCLSData(BaseDataSet):
  """mock data class for cls task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)

    samples_english = ["1\tAll is well", "0\tI am very angry"]
    samples_split_line_mark = ["1\t都挺好。|都是的呀", "0\t我很愤怒|超级生气！"]
    samples_split_by_space = ["1\t都 挺好", "0\t我 很 愤怒"]
    samples_split_by_char = ["1\t都挺好", "0\t我很愤怒"]
    samples_chinese_word = ["1\t都挺好", "0\t我很愤怒"]
    self.samples_dict = {"english": samples_english,
                    "split_by_line_mark": samples_split_line_mark,
                    "split_by_space": samples_split_by_space,
                    "split_by_char": samples_split_by_char,
                    "chinese_word": samples_chinese_word}

    self.train_file = "train.txt"
    self.dev_file = "dev.txt"
    self.test_file = "test.txt"
    self.text_vocab = "text_vocab.txt"
    files = [self.train_file, self.dev_file, self.test_file]
    self.data_files = [x.replace("txt", "")+data_type +".txt"
                       for x in files for data_type in self.samples_dict]
    self.config_files = ['cnn_cls_mock.yml']
    self.download_files = []

    text_vocab_english = ["<unk>\t0", "</s>\t1", "all\t3", "is\t4",
                          "well\t5", "i\t6", "am\t7", "very\t8"]
    text_vocab_split_line_mark = ["<unk>\t0", "</s>\t1", "都\t2", "挺好\t3",
                                  "我\t4", "很\t5", "|\t6", "是的\t7",
                                  "呀\t8", "超级\t9", "生气\t10"]
    text_vocab_split_by_space = ["<unk>\t0", "</s>\t1", "都\t2", "挺好\t3",
                                 "我\t4", "很\t5"]
    text_vocab_split_by_char = ["<unk>\t0", "</s>\t1", "都\t2", "挺\t3",
                                "好\t4", "我\t5", "很\t6", "愤\t7", "怒\t8"]
    text_vocab_chinese_word = ["<unk>\t0", "</s>\t1", "都\t2", "挺好\t3",
                               "我\t4", "很\t5"]
    self.text_vocab_dict = {"english": text_vocab_english,
                       "split_by_line_mark": text_vocab_split_line_mark,
                       "split_by_space": text_vocab_split_by_space,
                       "split_by_char": text_vocab_split_by_char,
                       "chinese_word": text_vocab_chinese_word}


  def download(self) -> bool:
    return True


  def after_download(self) -> bool:
    try:
      for data_type in self.samples_dict:

        samples = self.samples_dict[data_type]
        text_vocab_list = self.text_vocab_dict[data_type]

        train_file_path = os.path.join(self.data_dir,
                                       self.train_file.replace("txt", "") + data_type + ".txt")
        dev_file_path = os.path.join(self.data_dir,
                                     self.dev_file.replace("txt", "") + data_type + ".txt")
        test_file_path = os.path.join(self.data_dir,
                                      self.test_file.replace("txt", "") + data_type + ".txt")
        text_vocab_file = os.path.join(self.data_dir,
                                       self.text_vocab.replace("txt", "") + data_type + ".txt")

        mock_data(samples, train_file_path, dev_file_path, test_file_path, text_vocab_file, text_vocab_list)

    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
