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
'''Base class for Preparer'''

import os
import math
from pathlib import Path
from absl import logging
import delta.compat as tf
import numpy as np

from delta import utils
from delta.data.preprocess.utils import prepare_embedding
from delta.utils.solver.utils.solver_utils import get_session_conf
from delta.data.preprocess.utils import prepare_vocab
from delta.data.preprocess.utils import prepare_vocab_from_config
from delta.data.preprocess.utils import get_pre_process_text_ds_iter
from delta.data.utils.common_utils import get_file_len


class Preparer:
  '''Base class for Preparer'''

  def __init__(self, config):
    self.config = config
    self.reuse = self.config["data"]["task"]["preparer"].get("reuse", True)
    self.done_sign = self.config["data"]["task"]["preparer"].get(
        "done_sign", "")

  def skip_prepare(self):
    """Check if task need to skip the prepare process."""
    return self.done_sign != "" and os.path.exists(self.done_sign) \
           and self.reuse

  def done_prepare(self):
    """Touch a sign file after the prepare process is done."""
    if self.done_sign != "" and not os.path.exists(self.done_sign):
      if not os.path.exists(os.path.dirname(self.done_sign)):
        os.makedirs(os.path.dirname(self.done_sign))
      Path(self.done_sign).touch()

  def do_prepare(self, pre_process_pipeline):
    """Do the prepare processing."""
    raise NotImplementedError


class TextPreparer(Preparer):
  """Base Preparer class for nlp"""

  def __init__(self, config):
    super().__init__(config)
    self.all_modes = (utils.INFER, utils.EVAL, utils.TRAIN)
    self.infer_no_label = self.config["data"][utils.INFER].get(
        'infer_no_label', False)
    self.model_config = self.config["model"]
    self.task_config = self.config["data"]["task"]
    self.batch_size = self.task_config['batch_size']
    self.num_parallel_calls = self.task_config['num_parallel_calls']
    self.vocab_min_frequency = self.task_config['vocab_min_frequency']
    self.use_custom_vocab = self.task_config.get('use_custom_vocab', False)
    self.text_vocab_file_path = self.task_config['text_vocab']
    self.label_vocab_file_paths = self.task_config['label_vocab']
    if not isinstance(self.label_vocab_file_paths, list):
      self.label_vocab_file_paths = [self.label_vocab_file_paths]
    self.output_num = len(self.label_vocab_file_paths)
    self.multi_output = bool(self.output_num > 1)
    self.multi_text = False
    self.session_conf = get_session_conf(self.config)
    self.init_feed_dict = {}

  def prepare_raw_data(self, pre_process_pipeline):
    """
    Preparing raw data.
    For all kinds of text input, all_texts: [sentence1, ...]
    For single output, all_labels: [[label1, label2, ...]]
    For multiple outputs, all_labels: [[label1_1, ...], [label1_2, ...]]
    """
    if self.output_num <= 1:
      all_labels = []
    else:
      all_labels = [[] for _ in range(self.output_num)]
    all_texts = []
    for mode in self.all_modes:
      paths = self.config["data"][mode]['paths']
      paths_after_pre_process = [one_path + ".after" for one_path in paths]
      logging.debug(
          "paths_after_pre_process: {}".format(paths_after_pre_process))

      infer_without_label = bool(mode == utils.INFER and self.infer_no_label)

      for one_path, one_path_after in zip(paths, paths_after_pre_process):
        if not os.path.exists(one_path):
          raise FileNotFoundError("{} does not exist!".format(one_path))
        data_size = get_file_len([one_path])
        self.prepare_one_raw_data([one_path], one_path_after, mode,
                                  infer_without_label, pre_process_pipeline,
                                  all_texts, all_labels, data_size)
    if self.output_num <= 1:
      all_labels = [all_labels]
    return all_texts, all_labels

  def prepare_one_raw_data(self, one_path, one_path_after, mode,
                           infer_without_label, pre_process_pipeline, all_texts,
                           all_labels, data_size):
    """Prepare one raw data."""
    text, label = self.load_a_raw_file(one_path, infer_without_label)

    batch_num = int(math.ceil(data_size / float(self.batch_size)))
    if self.multi_text:
      one_text_after = []
      for i, one_text in enumerate(text):   #to be confirmed
        one_text_iterator = get_pre_process_text_ds_iter(
            one_text, pre_process_pipeline, self.num_parallel_calls,
            self.batch_size)
        text_after_arr = self.run_dataset(one_text_iterator,batch_num)
        text_after = [one_line.decode("utf-8") for one_line in text_after_arr]
        all_texts += text_after
        one_text_after.append(text_after)
    else:
      text = text[0]
      text_iterator = get_pre_process_text_ds_iter(text,
                                                   pre_process_pipeline,
                                                   self.num_parallel_calls,
                                                   self.batch_size)
      text_after_arr = self.run_dataset(text_iterator, batch_num)
      text_after = [one_line.decode("utf-8") for one_line in text_after_arr]
      all_texts += text_after
      one_text_after = text_after
    self.config['data']['{}_data_size'.format(mode)] = len(one_text_after[0])
    one_label_after = []
    if not infer_without_label:
      if self.multi_output:
        for i in range(self.output_num):
          label_ds = label[i].batch(self.batch_size)
          label_iterator = label_ds.make_initializable_iterator()
          label_after_arr = self.run_dataset(label_iterator, batch_num)
          label_after_one = [one_line.decode("utf-8") for one_line in label_after_arr]
          one_label_after.append(label_after_one)
          all_labels[i] += label_after_one
      else:
        label = label[0]
        label_ds = label.batch(self.batch_size)
        label_iterator = label_ds.make_initializable_iterator()
        label_after_arr = self.run_dataset(label_iterator, batch_num)
        one_label_after = [one_line.decode("utf-8") for one_line in label_after_arr]
        all_labels += one_label_after

    logging.debug(f"one_text_after: {len(one_text_after)}")
    self.save_a_raw_file(one_label_after, one_text_after, one_path_after,
                         infer_without_label)

  def run_dataset(self, data_iterator,batch_num):
    """Run the text pre-process pipeline, fetch data in numpy array format."""
    data_after = []
    data_t = data_iterator.get_next()
    with tf.Session(config=self.session_conf) as sess:
      sess.run(data_iterator.initializer, feed_dict=self.init_feed_dict)
      for _ in range(batch_num):
        try:
          data_after.append(sess.run(data_t))
        except tf.errors.OutOfRangeError:
          break
    data_after_arr = np.concatenate(data_after, axis=0)
    return data_after_arr


  def load_a_raw_file(self, one_path, infer_without_label):
    """
    Load a raw file. Return text and label.
    For single text input, text: [sentence1, ...]
    For multiple text inputs, text: [[sentence1_1, ...], [sentence1_2, ...]]
    For single output, label: [label1, label2, ...]
    For multiple outputs, label: [[label1_1, ...], [label1_2, ...]]
    """
    raise NotImplementedError

  def save_a_raw_file(self, label, text_after, one_path_after,
                      infer_without_label):
    """Save a raw file."""
    raise NotImplementedError

  def prepare_embed(self):
    """Preparing embedding."""
    logging.info("Preparing embedding ...")
    if self.model_config["use_pre_train_emb"]:
      prepare_embedding(self.model_config["pre_train_emb_path"],
                        self.task_config["text_vocab"],
                        self.model_config["embedding_path"])

  def prepare_text_vocab(self, all_texts):
    """Preparing text vocab"""
    if os.path.exists(self.text_vocab_file_path) and \
      self.use_custom_vocab:
      logging.info("Reuse text vocab file: {}".format(
          self.text_vocab_file_path))
    else:
      prepare_vocab(
          self.text_vocab_file_path,
          all_texts,
          min_frequency=self.vocab_min_frequency)
      logging.info("Generate text vocab file: {}".format(
          self.text_vocab_file_path))

  def prepare_label_vocab(self, all_labels):
    """Prepare label vocab"""
    for i in range(self.output_num):
      if os.path.exists(self.label_vocab_file_paths[i]) and \
        self.use_custom_vocab:
        logging.info("Reuse label vocab file: {}".format(
            self.label_vocab_file_paths[i]))
      else:
        if "vocab" in self.config["data"]["task"]["classes"]:
          output_index = i if self.multi_output else None
          prepare_vocab_from_config(
              self.label_vocab_file_paths[i],
              self.config,
              output_index=output_index)
        else:
          prepare_vocab(
              self.label_vocab_file_paths[i],
              all_labels[i],
              min_frequency=1,
              use_default_dict=False)
        logging.info("Generate label vocab file: {}".format(
            self.label_vocab_file_paths[i]))

  def prepare_vocabs(self, all_texts, all_labels):
    """Preparing vocab for x."""
    logging.info("Preparing vocab for x ...")
    self.prepare_text_vocab(all_texts)
    logging.info("Preparing vocab for y ...")
    self.prepare_label_vocab(all_labels)

  def do_prepare(self, pre_process_pipeline):
    """Do the prepare processing."""
    all_texts, all_labels = self.prepare_raw_data(pre_process_pipeline)
    self.prepare_vocabs(all_texts, all_labels)
    self.prepare_embed()
