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
"""Common utilities for data related operations."""
# pylint: disable=invalid-name

import re
import json
import numpy as np
from absl import logging
import delta.compat as tf
import subprocess

from delta import utils



def input_fn(dataset, mode, batch_size, num_epoch=None):
  '''
  params: dataset, tf.data.Dataset
  params: mode, learning phase
  params: batch size
  params: num of epoch
  '''
  if mode == utils.TRAIN:
    _, num_gpus = utils.gpu_device_names()
    per_device_batch_size = utils.per_device_batch_size(batch_size, num_gpus)
  else:
    # using one device to eval or infer,
    # otherwise will drop reminder samples, e.g. 32 batch with 3 gpus
    per_device_batch_size = batch_size
    num_epoch = 1

  logging.info(
      "Learning Phase: {}, Total Batch size:{}, Per device batch size: {}"
      .format(mode, batch_size, per_device_batch_size))

  def _input_fn():
    return dataset(mode, per_device_batch_size, num_epoch)

  return _input_fn


class JsonNumpyEncoder(json.JSONEncoder):
  """JSONEncoder warpper for numpy data."""

  # pylint: disable=arguments-differ, method-hidden
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super().default(obj)


def clean_english_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from
  https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  # pylint: disable=anomalous-backslash-in-string
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def save_a_text_cls_file(label, texts_after, new_path, no_label):
  """Save a text classification data to a file."""
  logging.info("Saving processed file to: {}".format(new_path))
  with open(new_path, "w", encoding="utf-8") as out_f:
    for i, one_line in enumerate(texts_after):
      if no_label:
        out_f.write(one_line + "\n")
      else:
        out_f.write(label[i] + "\t" + one_line + "\n")


def save_a_text_match_file(label, texts_after, new_path, no_label):
  """Save a text match data to a file."""
  logging.info("Saving processed file to: {}".format(new_path))
  with open(new_path, "w", encoding="utf-8") as out_f:
    for i, (one_line_l, one_line_r) in enumerate(zip(*texts_after)):
      if no_label:
        out_f.write(one_line_l + '\t' + one_line_r + "\n")
      else:
        out_f.write(label[i] + "\t" + one_line_l + '\t' + one_line_r + "\n")


def save_a_text_seq_label_file(label, texts_after, new_path, no_label):
  """Save a text seqlabel data to a file."""
  logging.info("Saving processed file to: {}".format(new_path))
  with open(new_path, "w", encoding="utf-8") as out_f:
    for i, one_line in enumerate(texts_after):
      if no_label:
        out_f.write(one_line + "\n")
      else:
        out_f.write(label[i] + "\t" + one_line + "\n")


def save_a_text_seq2seq_file(texts_after, new_path):
  """Save a text sequence data to a file"""
  logging.info("Saving processed file to: {}".format(new_path))
  with open(new_path, "w", encoding="utf-8") as out_f:
    for _, one_line in enumerate(texts_after):
      out_f.write(one_line + "\n")


def save_a_text_nlu_joint_file(label, texts_after, new_path, no_label):
  """Save a text nlu joint data to a file."""
  intent_label, slots_label = label
  logging.info("Saving processed file to: {}".format(new_path))
  with open(new_path, "w", encoding="utf-8") as out_f:
    for i, one_line in enumerate(texts_after):
      if no_label:
        out_f.write(one_line + "\n")
      else:
        out_f.write(intent_label[i] + "\t" + slots_label[i] + "\t" + one_line +
                    "\n")


def load_npy(npy_path, dtype=np.float32):
  """Load a data in npy format."""
  dense_feature = np.load(npy_path).astype(dtype)
  return dense_feature


def get_file_len(fname_paths):
  len_res = []
  for fname in fname_paths:
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
      raise IOError(err)
    len_res.append(int(result.strip().split()[0]))

  return sum(len_res)

def read_lines_from_text_file(file_path):
  """Read lines from a text file."""
  with open(file_path) as f:
    lines = [line.strip() for line in f.readlines()]
    return lines
