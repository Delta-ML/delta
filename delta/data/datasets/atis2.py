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

import os
from absl import logging
import shutil
import traceback
from delta.data.datasets.base_dataset import BaseDataSet
from delta.utils.register import registers


@registers.dataset.register('atis2')
class ATIS2(BaseDataSet):
  """atis2 data class for nlu joint task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.dev_file = "dev.txt"
    self.test_file = "test.txt"
    self.data_files = [self.train_file, self.dev_file, self.test_file]
    self.train_download = "origin_data/atis-2.train.w-intent.iob"
    self.dev_download = "origin_data/atis-2.dev.w-intent.iob"
    self.test_download = "origin_data/atis.test.w-intent.iob"
    self.download_files = [self.train_download, self.dev_download, self.test_download]
    self.config_files = ['atis2_nlu_joint_lstm_crf.yml']

  @staticmethod
  def to_standard_format(input_file, output_file):
    """change data format for data input"""
    logging.info("Save file to {}".format(output_file))

    with open(input_file, encoding="utf-8") as in_file, \
      open(output_file, "w", encoding="utf-8") as out_file:
      for row in in_file:
        parts = row.strip().split("\t")
        if len(parts) < 2:
          continue
        text = parts[0]
        sub_parts = parts[1].split(" ")
        intent_label = sub_parts[-1]
        slots_label = " ".join(sub_parts[:-1])

        text = text.rstrip("EOS")
        text = text.strip()

        out_file.write(intent_label + "\t"
                       + slots_label + "\t"
                       + text + "\n")

  def download(self) -> bool:
      github_url = "https://github.com/yvchen/JointSLU.git"
      res = os.system(f'cd {self.download_dir}; git clone {github_url}')
      if res != 0:
        return False
      return True

  def after_download(self) -> bool:
    try:
      shutil.move(os.path.join(self.download_dir, "JointSLU/data"),
                  os.path.join(self.download_dir, "origin_data"))
      shutil.rmtree(os.path.join(self.download_dir, "JointSLU"))
      self.to_standard_format(os.path.join(self.download_dir, self.train_download),
                              os.path.join(self.data_dir, self.train_file))
      self.to_standard_format(os.path.join(self.download_dir, self.dev_download),
                              os.path.join(self.data_dir, self.dev_file))
      self.to_standard_format(os.path.join(self.download_dir, self.test_download),
                              os.path.join(self.data_dir, self.test_file))
    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
