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

def generate_vocab(train_path, raw_vocab_path, new_vocab_path, label_vocab_path):
  f1 = open(new_vocab_path, 'w')
  with open(raw_vocab_path, 'r') as f:
    for idx, line in enumerate(f.readlines()):
      word = line.strip()
      if word == '[UNK]':
        word = '<unk>'
      f1.write(word + '\t' + str(idx) + '\n')
  label_vocab_file = open(label_vocab_path, 'w')
  label_vocab = {}
  with open(train_path, 'r') as f:
    for line in f.readlines():
      labels = line.strip().split('\t')[0]
      for t in labels.split(' '):
        if t in label_vocab:
          label_vocab[t] += 1
        else:
          label_vocab[t] = 1
  label_vocab = sorted(label_vocab.items(), key=lambda x: x[1], reverse=True)
  idx = 0
  for label, count in label_vocab:
    label_vocab_file.write(label + '\t' + str(idx) + '\n')
    idx += 1

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 5:
    logging.error("Usage python {} train_path raw_vocab_path, new_vocab_path, "
                  "label_vocab_path".format(sys.argv[0]))
    sys.exit(-1)

  train_path = sys.argv[1]
  raw_vocab_path = sys.argv[2]
  new_vocab_path = sys.argv[3]
  label_vocab_path = sys.argv[4]
  generate_vocab(train_path, raw_vocab_path, new_vocab_path, label_vocab_path)
