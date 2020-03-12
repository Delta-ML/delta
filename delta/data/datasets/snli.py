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

Stanford Natural Language Inference (SNLI) corpus is released in A large annotated
corpus for learning natural language inference

Available: https://sigann.github.io/LAW-XI-2017/papers/LAW01.pdf

## Download Links

https://nlp.stanford.edu/projects/snli/snli_1.0.zip

## Description

Stanford Natural Language Inference corpus is a new, freely available collection of
labeled sentence pairs, written by humans doing a novel grounded task based on image captioning.
At 570K pairs, it is two orders of magnitude larger than all other resources of its type.
This in- crease in scale allows lexicalized classi- fiers to outperform some sophisticated
existing entailment models, and it allows a neural network-based model to perform competitively
on natural language infer- ence benchmarks for the first time.

## Data scale introduction

- Training pairs：550,152
- Development pairs：10,000
- Test pairs：10,000
"""

import wget
import os
import traceback
import json
from absl import logging
from delta.data.datasets.base_dataset import BaseDataSet
from delta.utils.register import registers


@registers.dataset.register('snli')
class SNLI(BaseDataSet):
  """snli data class test for match task."""

  def __init__(self, project_dir):
    super().__init__(project_dir)
    self.train_file = "train.txt"
    self.dev_file = "dev.txt"
    self.test_file = "test.txt"
    self.data_files = [self.train_file, self.test_file, self.dev_file]
    self.config_files = ["snli_match_rnn.yml"]
    self.download_files = ["snli_1.0.zip"]

  def download(self) -> bool:
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    try:
      wget.download(url, self.download_dir)
    except Exception as e:
      logging.warning(repr(e))
      return False
    return True

  @staticmethod
  def to_standard_format(input_file, output_file):

    label_dic = {"neutral": "2", "contradiction": "0", "entailment": "1"}
    with open(input_file, encoding="utf-8") as json_file, \
      open(output_file, "w", encoding="utf-8") as out_file:
      text_reader = json_file.readlines()
      for line in text_reader:
        line_dic = json.loads(line)
        if "gold_label" not in line_dic or "sentence1" not in line_dic or "sentence2" not in line_dic:
          continue
        if line_dic["gold_label"] == "-":
          continue
        label = label_dic[line_dic["gold_label"]]
        sentence1 = line_dic["sentence1"]
        sentence2 = line_dic["sentence2"]
        out_file.write(label + "\t" + sentence1 + "\t" + sentence2 + "\n")

  def after_download(self) -> bool:
    try:
      download_file = os.path.join(self.download_dir, "snli_1.0.zip")
      os.system(f"unzip {download_file}  -d {self.download_dir}")
      self.to_standard_format(os.path.join(self.download_dir, "snli_1.0/snli_1.0_train.jsonl"),
                              os.path.join(self.data_dir, self.train_file))
      self.to_standard_format(os.path.join(self.download_dir, "snli_1.0/snli_1.0_dev.jsonl"),
                              os.path.join(self.data_dir, self.dev_file))
      self.to_standard_format(os.path.join(self.download_dir, "snli_1.0/snli_1.0_test.jsonl"),
                              os.path.join(self.data_dir, self.test_file))
    except Exception as e:
      logging.warning(traceback.format_exc())
      return False
    return True
