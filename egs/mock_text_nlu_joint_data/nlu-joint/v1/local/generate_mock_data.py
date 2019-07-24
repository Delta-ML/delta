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

import sys
from absl import logging
from delta.data.utils.test_utils import mock_a_text_file
from delta.data.utils.test_utils import save_a_vocab_file

# samples with label
samples = ["0\tO O O\t 我 心情 低落",
           "1\tO O B-ORG O O O\t我 在 肯德基 玩的 很 开心"]

text_vocab_list = ["<unk>\t0", "</s>\t1", "我\t2", "在\t3",
                   "肯德基\t4", "玩的\t5", "很\t6", "开心\t7"]


def mock_text_class_data(train_file, dev_file, test_file, text_vocab_file):
  logging.info("Generate mock data: {}".format(train_file))
  mock_a_text_file(samples, 500, train_file)
  logging.info("Generate mock data: {}".format(dev_file))
  mock_a_text_file(samples, 100, dev_file)
  logging.info("Generate mock data: {}".format(test_file))
  mock_a_text_file(samples, 100, test_file)
  logging.info("Generate text vocab file: {}".format(text_vocab_file))
  save_a_vocab_file(text_vocab_file, text_vocab_list)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 5:
    logging.error("Usage python {} train_file dev_file test_file text_vocab_file".format(sys.argv[0]))
    sys.exit(-1)

  train_file = sys.argv[1]
  dev_file = sys.argv[2]
  test_file = sys.argv[3]
  text_vocab_file = sys.argv[4]

  mock_text_class_data(train_file, dev_file, test_file, text_vocab_file)
