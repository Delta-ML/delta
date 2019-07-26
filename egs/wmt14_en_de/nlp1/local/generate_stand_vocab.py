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

def generate_stand_vocab(old_vocab, new_vocab):
  vocab_file = open(new_vocab, 'w')
  vocab_file.write('<pad>' + '\t' + '0' + '\n')
  vocab_file.write('<s>' + '\t' + '1' + '\n')
  vocab_file.write('</s>' + '\t' + '2' + '\n')
  vocab_file.write('<unk>' + '\t' + '3' + '\n')
  vocab_file.write('<sos>' + '\t' + '4' + '\n')
  vocab_file.write('<eos>' + '\t' + '5' + '\n')
  idx = 6
  with open(old_vocab, 'r') as f:
    for i, line in enumerate(f.readlines()):
      if i > 2:
        vocab_file.write(line.strip() + '\t' +
                         str(idx) + '\n')

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 3:
    logging.error("Usage python {} old_vocab new_vocab".format(sys.argv[0]))
    sys.exit(-1)

  old_vocab = sys.argv[1]
  new_vocab = sys.argv[2]
  generate_stand_vocab(old_vocab, new_vocab)


