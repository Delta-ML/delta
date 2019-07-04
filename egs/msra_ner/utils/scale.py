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
If the scale rate is smaller than 1, it returns the subset of the origin data;
If the scale rate is bigger than 1, it repeats the data and return.
"""

import sys
from absl import logging


def data_generator(data):
  while True:
    for i in range(len(data)):
      yield data[i]


def scale_data(original_file, new_file, scale_rate):
  logging.info("Scale file from {} to {}".format(original_file, new_file))

  with open(original_file, encoding="utf-8") as original_f, \
    open(new_file, "w", encoding="utf-8") as new_f:

    original_lines = original_f.readlines()
    original_size = len(original_lines)
    new_size = int(original_size * scale_rate)

    for i, line in enumerate(data_generator(original_lines)):
      if i >= new_size:
        break
      new_f.write(line)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 4:
    logging.error("Usage python {} original_file new_file scale_rate".format(
        sys.argv[0]))
    sys.exit(-1)

  original_file = sys.argv[1]
  new_file = sys.argv[2]
  scale_rate = float(sys.argv[3])

  scale_data(original_file, new_file, scale_rate)
