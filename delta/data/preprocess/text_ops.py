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

import tensorflow as tf
from absl import logging

from delta import utils
from delta.layers.ops import py_x_ops


def tokenize_label(label, maxlen, label_vocab_file_path, pad_id):
  """Tokenize labels"""
  label_id, _ = py_x_ops.sentence_to_ids(
      label,
      maxlen=maxlen,
      vocab_filepath=label_vocab_file_path,
      load_token_ids_from_vocab=True,
      pad_id=pad_id,
      check_tokens=False)
  return label_id


def tokenize_sentence(texts, max_seq_len, vocab_path):
  """Tokenize sentence"""
  token_ids, _ = py_x_ops.sentence_to_ids(
      texts,
      maxlen=max_seq_len,
      vocab_filepath=vocab_path,
      load_token_ids_from_vocab=True,
      pad_id=utils.PAD_IDX,
      check_tokens=False)
  return token_ids


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

