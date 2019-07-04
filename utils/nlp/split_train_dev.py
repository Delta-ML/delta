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
from sklearn.model_selection import train_test_split
from absl import logging


def split_train_dev(original_train, train_file, dev_file, split_rate):
  with open(original_train, encoding="utf-8") as original_train_f, \
    open(train_file, "w", encoding="utf-8") as train_f, \
    open(dev_file, "w", encoding="utf-8") as dev_f:
    lines = original_train_f.readlines()
    lines_train, lines_dev = train_test_split(
        lines, test_size=split_rate, random_state=2019)

    logging.info("Save train file to {}".format(train_file))
    for line in lines_train:
      train_f.write(line)

    logging.info("Save train file to {}".format(dev_file))
    for line in lines_dev:
      dev_f.write(line)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 5:
    logging.error(
        "Usage python {} original_train train_file dev_file split_rate".format(
            sys.argv[0]))
    sys.exit(-1)

  original_train = sys.argv[1]
  train_file = sys.argv[2]
  dev_file = sys.argv[3]
  split_rate = float(sys.argv[4])

  split_train_dev(original_train, train_file, dev_file, split_rate)
