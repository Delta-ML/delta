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


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  if len(sys.argv) != 3:
    logging.error("Usage {} input_file output_file".format(sys.argv[0]))
    sys.exit(-1)

  input_file = sys.argv[1]
  output_file = sys.argv[2]
  to_standard_format(input_file, output_file)
