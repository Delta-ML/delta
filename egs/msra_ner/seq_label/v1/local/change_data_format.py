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


def change_data_format(files):
  for data_file_in in files:
    if data_file_in == sys.argv[3]:
      logging.info("Change data format: {}".format(data_file_in))
      data_file_out = data_file_in.replace(".in", ".out")
      with open(data_file_out, "w", encoding="utf-8") as output_file:
        with open(data_file_in, "r", encoding="utf-8") as file_input:
          for line in file_input.readlines():
            word = list(line.strip())
            if len(line.strip()) != 0:
              output_file.write(' '.join(word) + "\n")
      return

    logging.info("Change data format: {}".format(data_file_in))
    data_file_out = data_file_in.replace(".in", ".out")
    words, labels = [], []
    with open(data_file_out, "w", encoding="utf-8") as output_file:
      with open(data_file_in, "r", encoding="utf-8") as file_input:
        for line in file_input.readlines():
          word = line.strip().split('\t')[0]
          label = line.strip().split('\t')[-1]
          # here we dont do "DOCSTART" check
          if len(line.strip()) == 0:
            l = [label for label in labels if len(label) > 0]
            w = [word for word in words if len(word) > 0]
            assert len(l) == len(w)
            l, w = ' '.join(l), ' '.join(w)
            output_file.write(l + "\t" + w + "\n")
            words, labels = [], []
          words.append(word)
          labels.append(label)
    logging.info("Change data done: {}".format(data_file_out))



if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 4:
    logging.error("Usage python {} train_file, dev_file, test_file".format(sys.argv[0]))
    sys.exit(-1)

  train_file = sys.argv[1]
  dev_file = sys.argv[2]
  test_file = sys.argv[3]
  files = [train_file, dev_file, test_file]

  change_data_format(files)
