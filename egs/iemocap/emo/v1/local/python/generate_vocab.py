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
'''generate vocab and embed for text'''

import sys
import os
import re
import numpy as np
from absl import logging
from collections import defaultdict
import pickle

def generate_vocab(data_path, text_vocab_path, min_freq):
  text_vocab = defaultdict(int)
  for parent, _, filenames in os.walk(data_path):
    for name in filenames:
        if name.endswith(".txt"):
          with open(os.path.join(parent, name), "r") as f:
            for line in f.readlines():
              text = re.findall(r"[\w']+|[.,!?;]", line.strip().lower())
              for w in text:
                text_vocab[w] += 1
  text_vocab_file = open(text_vocab_path, 'w')
  sorted_text_vocab = sorted(text_vocab.items(), key=lambda x: x[1], reverse=True)
  text_vocab_file.write('<pad>' + '\t' + str(0) + '\n')
  text_vocab_file.write('<unk>' + '\t' + str(1) + '\n')
  idx = 2
  text_vocab = {'<pad>': 0, '<unk>': 1}
  for word, count in sorted_text_vocab:
    if count >= min_freq:
      text_vocab_file.write(word + '\t' + str(idx) + '\n')
      text_vocab[word] = idx
      idx += 1
  try:
    assert idx == len(text_vocab)
  except Exception as e:
    print("idx:", idx)
    print("vocab:", len(text_vocab))

  logging.info("generate vocab: {}".format(idx))
  return text_vocab


def generate_embed(embed_path, text_vocab,
                   embed_pickle_path, embed_size=300):
  embeddings_dict = {}
  with open(embed_path, encoding="utf-8") as f:
    for line in f:
      try:
        values = line.replace("\n", "").split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = coefs
      except Exception as e:
        print("\tException: ", e)
    embedding_matrix = np.zeros((len(text_vocab), embed_size))

    for word, i in text_vocab.items():
        if word in embeddings_dict:
            embedding_matrix[int(i)] = embeddings_dict[word]
        else:
            embedding_matrix[int(i)] = np.random.uniform(-1, 1, embed_size)
            logging.info("out_of_vocabs {}".format(word))
    with open(embed_pickle_path, 'ab+') as fp:
      pickle.dump(embedding_matrix, fp)
    logging.info("embedding_matrix: {}".format(embedding_matrix.shape))
    logging.info("finish generate embed")

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 6:
    logging.error("Usage python {} data_path, text_vocab_path, "
                  "min_freq, embed_path, embed_pickle_path".format(sys.argv[0]))
    sys.exit(-1)

  data_path = sys.argv[1]
  text_vocab_path = sys.argv[2]
  min_freq = int(sys.argv[3])
  embed_path = sys.argv[4]
  embed_pickle_path = sys.argv[5]
  text_vocab = generate_vocab(data_path, text_vocab_path, min_freq)
  generate_embed(embed_path, text_vocab, embed_pickle_path)

