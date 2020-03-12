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

import os
import pickle
import numpy as np
from pathlib import Path
import tempfile
from absl import logging
from sklearn.model_selection import train_test_split


def data_generator(data):
  """Simple data generator"""
  while True:
    for i, ele in enumerate(data):
      yield data[i]


def scale_data(original_file, new_file, scale_rate):
  """Scale data file."""
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


def split_train_dev(original_train, train_file, dev_file, split_rate):
  """Split train and dev data set."""
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


def summary_joint_nlu_data(fname, output_file_path):
  with open(fname, 'rb') as stream:
    ds, dicts = pickle.load(stream)
  logging.info('      samples: {:4d}'.format(len(ds['query'])))
  logging.info('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
  logging.info('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
  logging.info(' intent count: {:4d}'.format(len(dicts['intent_ids'])))

  t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
  i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])
  query, slots, intent = map(ds.get,
                             ['query', 'slot_labels', 'intent_labels'])

  with open(output_file_path, "w", encoding="utf-8") as out_file:
    for i in range(len(query)):
      out_file.write(i2in[intent[i][0]] + "\t")
      out_file.write(' '.join(map(i2s.get, slots[i])) + "\t")
      out_file.write(' '.join(map(i2t.get, query[i])) + "\n")


def random_upsampling(data, sample_num):
  """Up sample"""
  np.random.seed(2019)
  new_indices = np.random.permutation(range(sample_num))
  original_size = len(data)
  new_data = [data[i % original_size] for i in new_indices]
  return new_data


def mock_a_text_file(sample_lines, line_num, file_name):
  """Generate a mock text file for test."""
  with open(file_name, "w", encoding="utf-8") as f:
    lines = random_upsampling(sample_lines, line_num)
    for line in lines:
      f.write(line + "\n")


def generate_vocab_file():
  """Generate Vocab file for test. no usage now """
  tmpdir = Path(tempfile.mkdtemp())
  vocab_file = str(tmpdir.joinpath('vocab.txt'))
  dummy_vocabs = ["</s>", "<unk>", "你好", "北京"]
  save_a_vocab_file(vocab_file, dummy_vocabs)
  return vocab_file


def save_a_vocab_file(vocab_file, vocab_list):
  """Save a Vocab file for test."""
  with open(vocab_file, "w", encoding='utf-8') as out_f:
    for vocab in vocab_list:
      out_f.write(vocab)
      out_f.write('\n')
  return vocab_file


def split_file(ori_file):
  src_file = ori_file + '.src'
  tgt_file = ori_file + '.tgt'
  with open(ori_file, 'r', encoding='utf8') as in_f:
    lines = in_f.readlines()
  src, tgt = zip(*[sent.split('\t') for sent in lines])
  with open(src_file, 'w', encoding='utf8') as out_f:
    for src_sent in src:
      out_f.write(src_sent+'\n')
  with open(tgt_file, 'w', encoding='utf8') as out_f:
    for tgt_sent in tgt:
      out_f.write(tgt_sent)
  os.remove(ori_file)


def mock_data(samples, train_file, dev_file, test_file, text_vocab_file=None,
              text_vocab_list=None, label_vocab_file=None, label_vocab_list=None):
  logging.info("Generate mock data: {}".format(train_file))
  mock_a_text_file(samples, 300, train_file)
  logging.info("Generate mock data: {}".format(dev_file))
  mock_a_text_file(samples, 100, dev_file)
  logging.info("Generate mock data: {}".format(test_file))
  mock_a_text_file(samples, 100, test_file)

  if text_vocab_file and text_vocab_list:
    logging.info("Generate text vocab file: {}".format(text_vocab_file))
    save_a_vocab_file(text_vocab_file, text_vocab_list)
  if label_vocab_file and label_vocab_file:
    logging.info("Generate label vocab file: {}".format(label_vocab_file))
    save_a_vocab_file(label_vocab_file, label_vocab_list)

  # for seq2seq
  if not text_vocab_file and not text_vocab_list and not \
    label_vocab_file and not label_vocab_list:
    logging.info("Generate mock data: {} and split to src and tgt.".format(train_file))
    split_file(train_file)
    logging.info("Generate mock data: {} and split to src and tgt.".format(dev_file))
    split_file(dev_file)
    logging.info("Generate mock data: {} and split to src and tgt.".format(test_file))
    split_file(test_file)
