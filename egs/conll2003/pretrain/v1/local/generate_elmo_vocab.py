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

def generate_vocab(data_path, text_vocab_path,
                   label_vocab_path, min_freq):
  text_vocab = {}
  label_vocab = {}
  file_list = ["train.txt", "dev.txt", "test.txt"]
  for f in file_list:
    file_name = data_path + f
    with open(file_name, 'r') as f:
      for line in f.readlines():
        text = line.strip().split('\t')[1]
        labels = line.strip().split('\t')[0]
        for w in text.split(' '):
          if w in text_vocab:
            text_vocab[w] += 1
          else:
            text_vocab[w] = 1
        for t in labels.split(' '):
          if t in label_vocab:
            label_vocab[t] += 1
          else:
            label_vocab[t] = 1
  text_vocab_file = open(text_vocab_path, 'w')
  label_vocab_file = open(label_vocab_path, 'w')
  text_vocab_file.write('<pad>' + '\t' + '0' + '\n')
  text_vocab_file.write('<s>' + '\t' + '1' + '\n')
  text_vocab_file.write('</s>' + '\t' + '2' + '\n')
  text_vocab_file.write('<unk>' + '\t' + '3' + '\n')
  text_vocab_file.write('<sos>' + '\t' + '4' + '\n')
  text_vocab_file.write('<eos>' + '\t' + '5' + '\n')
  text_vocab = sorted(text_vocab.items(), key=lambda x: x[1], reverse=True)
  label_vocab = sorted(label_vocab.items(), key=lambda x: x[1], reverse=True)
  idx = 6
  for word, count in text_vocab:
    if count >= min_freq:
      text_vocab_file.write(word + '\t' + str(idx) + '\n')
      idx += 1
  idx = 0
  for label, count in label_vocab:
    label_vocab_file.write(label + '\t' + str(idx) + '\n')
    idx += 1
  logging.info("finish generate vocab!")

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 5:
    logging.error("Usage python {} data_path, text_vocab_path, label_vocab_path, "
                  "min_freq".format(sys.argv[0]))
    sys.exit(-1)

  data_path = sys.argv[1]
  text_vocab_path = sys.argv[2]
  label_vocab_path = sys.argv[3]
  min_freq = int(sys.argv[4])
  generate_vocab(data_path, text_vocab_path,
                 label_vocab_path, min_freq)



