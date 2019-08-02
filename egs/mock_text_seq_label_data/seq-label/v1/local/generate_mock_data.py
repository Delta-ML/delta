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
samples = ["O O O O\tI feel good .",
           "O O B-ORG O O O O O O\tBy stumps Kent had reached 108 for three ."]
text_vocab_list = ["<unk>\t0", "</s>\t1", "I\t2", "feel\t3","good\t4", ".\t5", 
                   "By\t6", "stumps\t7", "Kent\t8", "had\t9", "reached\t10", "108\t11", "for\t12", "three\t13"]
label_vocab_list = ["O\t0", "B-PER\t1", "I-PER\t2", "B-LOC\t3", "I-LOC\t4",
                    "B-ORG\t5", "I-ORG\t6", "B-MISC\t7", "I-MISC\t8"]


def mock_text_class_data(train_file, dev_file, test_file, text_vocab_file, label_vocab_file):
  logging.info("Generate mock data: {}".format(train_file))
  mock_a_text_file(samples, 500, train_file)
  logging.info("Generate mock data: {}".format(dev_file))
  mock_a_text_file(samples, 100, dev_file)
  logging.info("Generate mock data: {}".format(test_file))
  mock_a_text_file(samples, 100, test_file)
  logging.info("Generate text vocab file: {}".format(text_vocab_file))
  save_a_vocab_file(text_vocab_file, text_vocab_list)
  logging.info("Generate label vocab file: {}".format(label_vocab_file))
  save_a_vocab_file(label_vocab_file, label_vocab_list)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 6:
    logging.error("Usage python {} train_file dev_file test_file text_vocab_file".format(sys.argv[0]))
    sys.exit(-1)

  train_file = sys.argv[1]
  dev_file = sys.argv[2]
  test_file = sys.argv[3]
  text_vocab_file = sys.argv[4]
  label_vocab_file = sys.argv[5]


  mock_text_class_data(train_file, dev_file, test_file, text_vocab_file, label_vocab_file)
