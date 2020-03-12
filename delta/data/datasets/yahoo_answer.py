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
## Description
Yahoo answers are obtained from (Zhang et al., 2015). This is a topic classification task with 10 classes: Society & Culture,
Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music,
Family & Relationships and Politics & Government. The document we use includes question titles, question contexts and best answers.

## Download links

https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tg

## Data scale introduction

we split the raw data into training set, development dataset and test dataset
- Training dataset：1260,000
- Development pairs：140,000
- Test pairs：60,000
"""

import wget
import os
import traceback
import csv
from absl import logging
from delta.data.datasets.base_dataset import BaseDataSet
from delta.utils.register import registers
from delta.data.datasets.utils import split_train_dev


@registers.dataset.register('yahoo_answer')
class YahooAnswer(BaseDataSet):
  """yahoo answer data class for cls task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.data_files = ["train.txt", "test.txt", "dev.txt"]
    self.config_files = ['yahoo_answer_text_cls_han.yml']
    self.download_files = ["yahoo_answers_csv.tgz"]

  def download(self) -> bool:
    url = "https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz"
    try:
      wget.download(url, self.download_dir)
    except Exception as e:
      logging.warning(repr(e))
      return False
    return True

  @staticmethod
  def to_standard_format(input_file, output_file):
    logging.info("Save file to {}".format(output_file))

    with open(input_file, encoding="utf-8") as csv_file, \
      open(output_file, "w", encoding="utf-8") as out_file:
      csv_reader = csv.reader(csv_file)
      for row in csv_reader:
        if len(row) < 4:
          continue
        label = row[0]
        text = " ".join(row[1:])
        out_file.write(label + "\t" + text + "\n")

  def after_download(self) -> bool:
    try:
      download_file = os.path.join(self.download_dir, "yahoo_answers_csv.tgz")
      os.system(f"tar zxvf {download_file}  -C {self.download_dir}")
      self.to_standard_format(os.path.join(self.download_dir, "yahoo_answers_csv/train.csv"),
                              os.path.join(self.data_dir, "train_all.txt"))
      self.to_standard_format(os.path.join(self.download_dir, "yahoo_answers_csv/test.csv"),
                              os.path.join(self.data_dir, "test.txt"))
      split_train_dev(os.path.join(self.data_dir, "train_all.txt"),
                      os.path.join(self.data_dir, "train.txt"),
                      os.path.join(self.data_dir, "dev.txt"), 0.1)
    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
