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
## References

Xin Li, Dan Roth, Learning Question Classifiers. COLING'02, Aug., 2002.

## Description

This data collection contains all the data used in our learning question classification
experiments(Xin Li, Dan Roth, Learning Question Classifiers. COLING'02, Aug., 2002.),
which has question class definitions, the training and testing question sets, examples
of preprocessing the questions, feature definition scripts and examples of semantically
related word features.
This work has been done by Xin Li and Dan Roth and supported by Research supported by
(NSF grants IIS-9801638 and ITR IIS-0085836 and an ONR MURI Award.) .

## Download Links

https://github.com/thtrieu/qclass_dl/tree/master/data

## Data scale introduction

- Training size：5452
- Development size：-
- Test size：500

"""

import os
import traceback
import wget
from absl import logging
from delta.data.datasets.base_dataset import BaseDataSet
from delta.utils.register import registers


@registers.dataset.register('trec')
class TREC(BaseDataSet):
  """trec data class test for cls task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.test_file = "test.txt"
    self.data_files = [self.train_file, self.test_file]
    self.config_files = ['trec_text_cls_cnn.yml']
    self.download_files = ["train", "test"]

  def download(self) -> bool:
    train_url = "https://raw.githubusercontent.com/thtrieu/qclass_dl/master/data/train"
    test_url = "https://raw.githubusercontent.com/thtrieu/qclass_dl/master/data/test"
    try:
      wget.download(train_url, self.download_dir)
      wget.download(test_url, self.download_dir)
    except Exception as e:
      logging.warning(repr(e))
      return False
    return True

  @staticmethod
  def to_standard_format(input_file, output_file):
    logging.info("Save file to {}".format(output_file))

    max_seq = 0
    with open(input_file, encoding="ISO-8859-1") as in_file, \
      open(output_file, "w", encoding="utf-8") as out_file:
      for row in in_file.readlines():
        parts = row.strip().split(" ")
        label = parts[0].split(":")[0]
        text_len = len(parts[1:])
        if text_len > max_seq:
          max_seq = text_len
        text = " ".join(parts[1:])
        out_file.write(label + "\t" + text + "\n")
    logging.info("max seq len is {}".format(max_seq))

  def after_download(self) -> bool:
    try:
      self.to_standard_format(os.path.join(self.download_dir, "train"),
                              os.path.join(self.data_dir, self.train_file))
      self.to_standard_format(os.path.join(self.download_dir, "test"),
                              os.path.join(self.data_dir, self.test_file))
    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
