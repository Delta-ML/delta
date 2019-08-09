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
import sys
from absl import logging
from delta.data.utils.test_utils import mock_a_text_file

# samples with label
samples = [" a shooting at a bar popular with expatriates in mali on saturday killed "
           "five people \t killed five people",
           "a pennsylvania community is pulling together to search for an eighth-grade "
           "student who has been missing since wednesday\tsearch for missing student"]


def mock_text_class_data(train_file, dev_file, test_file):
  logging.info("Generate mock data: {}".format(train_file))
  mock_a_text_file(samples, 300, train_file)
  split_file(train_file)
  logging.info("Generate mock data: {}".format(dev_file))
  mock_a_text_file(samples, 100, dev_file)
  split_file(dev_file)
  logging.info("Generate mock data: {}".format(test_file))
  mock_a_text_file(samples, 100, test_file)
  split_file(test_file)


def split_file(ori_file):
  src_file = ori_file + '.src'
  tgt_file = ori_file + '.tgt'
  with open(ori_file, 'r', encoding='utf8') as f:
    lines = f.readlines()
  src, tgt = zip(*[sent.split('\t') for sent in lines])
  with open(src_file, 'w', encoding='utf8') as f:
    for src_sent in src:
      f.write(src_sent+'\n')
  with open(tgt_file, 'w', encoding='utf8') as f:
    for tgt_sent in tgt:
      f.write(tgt_sent)
  os.remove(ori_file)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 4:
    logging.error("Usage python {} train_file, dev_file, test_file".format(sys.argv[0]))
    sys.exit(-1)

  train_file = sys.argv[1]
  dev_file = sys.argv[2]
  test_file = sys.argv[3]

  mock_text_class_data(train_file, dev_file, test_file)
