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
'''Utilities for data preprocessing'''

import os
import pickle
import collections
import numpy as np
from absl import logging
import delta.compat as tf

from delta.data.utils.vocabulary import Vocabulary


def get_pre_process_text_ds_iter(
    text_ds,
    pipeline_func,
    num_parallel_calls,
    batch_size,
):
  """Get pre-process oprators."""

  text_ds = text_ds.map(pipeline_func, num_parallel_calls=num_parallel_calls)

  text_ds = text_ds.batch(batch_size)

  iterator = text_ds.make_initializable_iterator()

  return iterator


def process_vocab(vocab_file_path, data, vocab, min_frequency=0):
  """Process vocab"""
  for line in data:
    for word in line.split():
      vocab.add(word)
  if min_frequency > 0:
    vocab.trim(min_frequency)
  save_vocabs(vocab.mapping, vocab_file_path)


def save_vocabs(vocabs, vocab_file_path):
  """Save vocabs, vocab: {"word": 1, ...}"""
  logging.info("Saving vocab to {}".format(vocab_file_path))
  id_to_vocab = {v: k for k, v in vocabs.items()}
  ordered_vocabs = collections.OrderedDict()
  for _id in sorted(vocabs.values()):
    ordered_vocabs[id_to_vocab[_id]] = _id

  if os.path.isfile(vocab_file_path):
    os.remove(vocab_file_path)
  if not os.path.exists(os.path.dirname(vocab_file_path)):
    os.makedirs(os.path.dirname(vocab_file_path))

  with open(vocab_file_path, "w", encoding='utf-8') as out_f:
    for word, _id in ordered_vocabs.items():
      out_f.write("{}\t{}\n".format(word, _id))


def load_vocab_dict(vocab_file_path):
  """Load vocabs, vocab: {"word": 1, ...}"""
  logging.info("Loading vocab from {}".format(vocab_file_path))
  with open(vocab_file_path, encoding='utf-8') as in_f:
    vocabs = {}
    for line in in_f:
      parts = line.rstrip().split("\t")
      if len(parts) < 2:
        continue
      vocabs[parts[0]] = parts[1]
  logging.info("Loded {} vocabs from {}".format(len(vocabs), vocab_file_path))
  return vocabs


def get_vocab_size(vocab_file_path):
  """Get vocab size."""
  vocab_dict = load_vocab_dict(vocab_file_path)
  return len(vocab_dict)


def prepare_vocab(vocab_file_path, text, min_frequency=1,
                  use_default_dict=True):
  """Prepare vocab"""
  text_vocab = Vocabulary(use_default_dict=use_default_dict)
  process_vocab(vocab_file_path, text, text_vocab, min_frequency=min_frequency)


def prepare_vocab_from_config(vocab_file_path, config, output_index=None):
  """Prepare vocab from config."""
  if output_index is None:
    label_config = config["data"]["task"]["classes"]["vocab"]
  else:
    label_config = config["data"]["task"]["classes"][output_index]["vocab"]

  save_vocabs(label_config, vocab_file_path)


def prepare_embedding(pre_train_emb_path, text_vocab_path, embedding_path):
  """Prepare embedding"""
  # pylint: disable=too-many-locals

  if os.path.isfile(embedding_path):
    os.remove(embedding_path)

  # load pre_train model
  logging.info("Loading embedding from {}".format(pre_train_emb_path))
  emb_data = open(pre_train_emb_path, encoding='utf-8', mode='r')
  emb_dict = {}
  emb_size = 0
  line_num = 0
  for line in emb_data.readlines():
    line_num += 1
    if line_num == 1:
      continue
    line = line.strip().split(' ')
    word = line[0]
    vector = [float(i) for i in line[1:]]
    vector = vector / np.linalg.norm(vector)
    emb_dict[word] = vector
    emb_size = len(vector)
  logging.info("Load {} vectors".format(line_num))
  # load text vocab
  vocabs = load_vocab_dict(text_vocab_path)

  # get embedding vector for words in vocab
  vocab_size = len(vocabs)
  emb_list = [[]] * vocab_size
  bound = np.sqrt(1.0) / np.sqrt(vocab_size)
  count_exist = 0
  count_not_exist = 0
  word_id = 0

  for word in vocabs:
    try:
      word_vector = emb_dict[word]
    except Exception:  # pylint: disable=broad-except
      word_vector = None

    if word_vector is not None:
      emb_list[word_id] = word_vector
      count_exist += 1
    else:
      count_not_exist += 1
      emb_list[word_id] = np.random.uniform(-bound, bound, emb_size)
    word_id += 1

  emb_list = np.array(emb_list)
  logging.info("embedding exist : {}, embedding not exist : {}".format(
      count_exist, count_not_exist))
  logging.info("embedding exist dump to: {}".format(embedding_path))
  with open(embedding_path, mode='wb') as out_f:
    pickle.dump(emb_list, out_f)
