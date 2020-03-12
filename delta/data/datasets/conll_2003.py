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

https://www.aclweb.org/anthology/W03-0419

## links for download data

https://www.clips.uantwerpen.be/conll2003/ner/

## Description

The CoNLL-2003 named entity data consists of eight files covering two languages:
English and German1.
For each of the languages there is a training file, a development file,
a test file and a large file with unannotated data.
The learning methods were trained with the training data.
The development data could be used for tuning the parameters of the learning methods

## Data scale introduction

| English DataSet |  Articles | Sentences | Tokens |
|---|---|---|---|
| Training set |  946 | 14,987 | 203,621
| Development set | 216 | 3,466 | 51,362 |
| Test set | 231 | 3,684 | 46,435 |


| English DataSet |  LOC | MISC | ORG |PER|
|---|---|---|---|---|
| Training set |  7140 | 3438 | 6321 | 6600|
| Development set |1837 | 922 | 1341 |1842|
| Test set | 1668 |702| 1661|1617|

 The more details about Germanl Dataset is shown in paper.


"""

import os
import traceback
import wget
from absl import logging
from delta.data.datasets.base_dataset import BaseDataSet
from delta.utils.register import registers


@registers.dataset.register('conll_2003')
class Conll2003(BaseDataSet):
  """conll2003 data class for seqlabel task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.dev_file = "dev.txt"
    self.test_file = "test.txt"
    self.data_files = [self.train_file, self.test_file, self.dev_file]
    self.config_files = ["conll_2003_seq_label_bert.yml",
                         "conll_2003_seq_label_elmo.yml",
                         "conll_2003_seq_label_lstm_crf.yml"]
    self.download_files = [self.train_file, self.test_file, self.dev_file]

  def download(self) -> bool:
    train_url = "https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/train.txt"
    dev_url = "https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/dev.txt"
    test_url = "https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/test.txt"
    try:
      wget.download(train_url, self.download_dir)
      wget.download(dev_url, self.download_dir)
      wget.download(test_url, self.download_dir)
    except Exception as e:
      logging.warning(repr(e))
      return False
    return True

  @staticmethod
  def to_standard_format(input_file, output_file):

      logging.info("Change data format: {}".format(input_file))
      words, labels = [], []
      with open(output_file, "w", encoding="utf-8") as output_file:
        with open(input_file, "r", encoding="utf-8") as file_input:
          for line in file_input.readlines():
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            # here we dont do "DOCSTART" check
            if len(line.strip()) == 0:
              l = [label for label in labels if not label]
              w = [word for word in words if not word]
              assert len(l) == len(w)
              l, w = ' '.join(l), ' '.join(w)
              output_file.write(l + "\t" + w + "\n")
              words, labels = [], []
            words.append(word)
            labels.append(label)
      logging.info("Change data done: {}".format(output_file))

  def after_download(self) -> bool:
    try:
      download_file = os.path.join(self.download_dir, "yahoo_answers_csv.tgz")
      os.system(f"tar zxvf {download_file}  -C {self.download_dir}")
      self.to_standard_format(os.path.join(self.download_dir, self.train_file),
                              os.path.join(self.data_dir, self.train_file))
      self.to_standard_format(os.path.join(self.download_dir, self.dev_file),
                              os.path.join(self.data_dir, self.dev_file))
      self.to_standard_format(os.path.join(self.download_dir, self.test_file),
                              os.path.join(self.data_dir, self.test_file))
    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
