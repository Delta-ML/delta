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
''' Text related pre-process in ops.'''

import delta.compat as tf
from absl import logging

from delta import utils
from delta.layers.ops import py_x_ops
from delta.data.utils import read_lines_from_text_file


def tokenize_label(label, maxlen, label_vocab_file_path, pad_id):
  """Tokenize labels"""
  vocabs = read_lines_from_text_file(label_vocab_file_path)
  label_id, _ = py_x_ops.sentence_to_ids(
      label,
      maxlen=maxlen,
      use_vocab_file=False,
      vocab=vocabs,
      load_token_ids_from_vocab=True,
      pad_id=pad_id,
      check_tokens=False)
  return label_id


def tokenize_sentence(texts, max_seq_len, vocab_path):
  """Tokenize sentence"""
  vocabs = read_lines_from_text_file(vocab_path)
  token_ids, _ = py_x_ops.sentence_to_ids(
      texts,
      maxlen=max_seq_len,
      use_vocab_file=False,
      vocab=vocabs,
      load_token_ids_from_vocab=True,
      pad_id=utils.PAD_IDX,
      check_tokens=False)
  return token_ids


def chinese_word_cut_tf(input_str, use_file=False):
  """"""

  if use_file:
    output_str = py_x_ops.jieba_cut(
      input_str,
      use_file=True,
      hmm=True)
  else:
    output_str = py_x_ops.jieba_cut(
      input_str,
      use_file=False,
      hmm=True)
  return output_str


def clean_english_str_tf(input_str):
  """Clean English string with tensorflow oprations."""
  # pylint: disable=anomalous-backslash-in-string
  string = tf.regex_replace(input_str, r"[^A-Za-z0-9(),!?\'\`<>/]", " ")
  string = tf.regex_replace(string, "\'s", " \'s")
  string = tf.regex_replace(string, "\'ve", " \'ve")
  string = tf.regex_replace(string, "n\'t", " n\'t")
  string = tf.regex_replace(string, "\'re", " \'re")
  string = tf.regex_replace(string, "\'d", " \'d")
  string = tf.regex_replace(string, "\'ll", " \'ll")
  string = tf.regex_replace(string, ",", " , ")
  string = tf.regex_replace(string, "!", " ! ")
  string = tf.regex_replace(string, "\(", " ( ")
  string = tf.regex_replace(string, "\)", " ) ")
  string = tf.regex_replace(string, "\?", " ? ")
  string = tf.regex_replace(string, "\s{2,}", " ")
  string = tf.string_strip(string)
  string = py_x_ops.str_lower(string)
  return string


def char_cut_tf(input_str):
  """Cut sentence char by char with tensoflow operations."""
  input_str = tf.convert_to_tensor(input_str)
  rank = len(input_str.get_shape())
  if rank == 1:
    output_str = tf.strings.unicode_split(input_str,
                                          "UTF-8").to_tensor(default_value="")
    output_str = tf.strings.reduce_join(output_str, axis=1, separator=" ")
  elif rank == 0:
    output_str = tf.strings.unicode_split(input_str, "UTF-8")
    output_str = tf.strings.reduce_join(output_str, axis=0, separator=" ")
  else:
    logging.error("Please check the shape of input_str!")
    raise Exception("Error input shape for input_str.")
  output_str = tf.strings.strip(output_str)
  return output_str


def load_textline_dataset(paths, column_num):
  """Load raw data for text task."""
  ds = tf.data.TextLineDataset(paths)
  ds = ds.map(lambda x: tf.strings.split(x, sep="\t", result_type="RaggedTensor"))
  ds = ds.filter(lambda line: tf.equal(tf.size(line), column_num))
  ds_list = []
  for i in range(column_num):
    ds_list.append(ds.map(lambda x: x[i]))

  return tuple(ds_list)


def process_one_label_dataset(label_ds, config, output_index=None):
  """process one-label data set."""

  logging.info("Loading one label dataset...")
  num_parallel_calls = config["data"]["task"]["num_parallel_calls"]
  classes = config["data"]["task"]["classes"]
  if isinstance(classes, list):
    if output_index is None or output_index not in range(len(classes)):
      raise IndexError("output_index:{} not in the range of classes length: "
                       "{}!".format(output_index, len(classes)))
    num_classes = classes[output_index]["num_classes"]
    label_vocab_file_path = config["data"]["task"]["label_vocab"][output_index]
  else:
    num_classes = classes["num_classes"]
    label_vocab_file_path = config["data"]["task"]["label_vocab"]

  label_ds = label_ds.map(
    lambda x: tokenize_label(
      x, maxlen=1, label_vocab_file_path=label_vocab_file_path, pad_id=0),
    num_parallel_calls=num_parallel_calls)

  label_ds = label_ds.map(
    lambda l: tf.one_hot(l, num_classes, dtype=tf.int32),
    num_parallel_calls=num_parallel_calls)

  label_ds = label_ds.map(tf.squeeze, num_parallel_calls=num_parallel_calls)

  return label_ds


def process_multi_label_dataset(label_ds, config, output_index=None):
  """process multi-label data set."""
  logging.info("Loading multi label dataset...")
  label_vocab_file_path = config["data"]["task"]["label_vocab"]
  num_parallel_calls = config["data"]["task"]["num_parallel_calls"]
  max_seq_len = config["data"]["task"]["max_seq_len"]

  label_vocab_file_path = config["data"]["task"]["label_vocab"]
  if isinstance(label_vocab_file_path, list):
    if output_index is None or output_index not in range(
      len(label_vocab_file_path)):
      raise IndexError("output_index:{} not in the range of classes length: "
                       "{}!".format(output_index, len(label_vocab_file_path)))
    label_vocab_file_path = label_vocab_file_path[output_index]

  else:
    label_vocab_file_path = label_vocab_file_path

  label_ds = label_ds.map(
    lambda x: tokenize_label(
      x,
      maxlen=max_seq_len,
      label_vocab_file_path=label_vocab_file_path,
      pad_id=0),
    num_parallel_calls=num_parallel_calls)
  label_ds = label_ds.map(tf.squeeze, num_parallel_calls=num_parallel_calls)

  return label_ds


def load_dense_dataset(dense_feature):
  """Load dense data set"""
  dataset = tf.data.Dataset.from_tensor_slices(dense_feature)
  return dataset
