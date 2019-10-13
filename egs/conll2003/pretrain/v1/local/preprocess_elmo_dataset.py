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

def add_start_end_token(file):
  f1 = open(file + ".elmo", 'w')
  with open(file, 'r') as f:
    add_lines = ['O ' + line.strip().split('\t')[0] + ' O'
                 + '\t' + '<s> ' + line.strip().split('\t')[1] + ' </s>'
                 for line in f.readlines()]
    f1.write('\n'.join(add_lines))

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 4:
    logging.error("Usage python {} train_path, val_path, test_path".format(sys.argv[0]))
    sys.exit(-1)

  train_path = sys.argv[1]
  val_path = sys.argv[2]
  test_path = sys.argv[3]
  add_start_end_token(train_path)
  add_start_end_token(val_path)
  add_start_end_token(test_path)
