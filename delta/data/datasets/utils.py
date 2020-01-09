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
from absl import logging
import pickle

from sklearn.model_selection import train_test_split


def data_generator(data):
  """Simple data generator"""
  while True:
    for i in range(len(data)):
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
