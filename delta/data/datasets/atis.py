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

Charles T. Hemphill, John J. Godfrey, and George R. Doddington. 1990.
The ATIS spoken language systems pilot corpus.
In Proceedings of the DARPA Speech and Natural Language Workshop.
http://www.aclweb.org/anthology/ H90-1021.

## Download Links

https://github.com/howl-anderson/ATIS_dataset/raw/master/data/raw_data/ms-cntk-atis

## Description

the Air Travel Information System (ATIS) pilot corpus,
a corpus designed to measure progress in Spoken Language Systems
that include both a speech and natural language component.
This pilot marks the first full-scale attempt to collect such a corpus
and provides guidelines for future efforts.


## Data scale introduction

- Training size：4,978
- Development size：-
- Test size：893
- Intents：26
- Slots：129

"""

import os
import traceback
import wget
from absl import logging
from delta.data.datasets.base_dataset import BaseDataSet
from delta.data.datasets.utils import summary_joint_nlu_data
from delta.utils.register import registers


@registers.dataset.register('atis')
class ATIS(BaseDataSet):
  """atis data class for nlu joint task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.test_file = "test.txt"
    self.data_files = [self.train_file, self.test_file]
    self.config_files = ['atis_nlu_joint_lstm_crf.yml']
    self.download_files = ["atis.train.pkl", "atis.test.pkl"]

  def download(self) -> bool:
    train_url = "https://github.com/howl-anderson/ATIS_dataset/raw/master/" \
                "data/raw_data/ms-cntk-atis/atis.train.pkl"
    test_url = "https://github.com/howl-anderson/ATIS_dataset/raw/master/" \
               "data/raw_data/ms-cntk-atis/atis.test.pkl"
    try:
      wget.download(train_url, self.download_dir)
      wget.download(test_url, self.download_dir)
    except Exception as e:
      logging.warning(repr(e))
      return False
    return True

  def after_download(self) -> bool:
    try:
      summary_joint_nlu_data(os.path.join(self.download_dir, "atis.train.pkl"),
                             os.path.join(self.data_dir, self.train_file))
      summary_joint_nlu_data(os.path.join(self.download_dir, "atis.test.pkl"),
                             os.path.join(self.data_dir, self.test_file))
    except Exception as e:

      logging.warning(traceback.format_exc())
      return False
    return True
