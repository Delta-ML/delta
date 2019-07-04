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
''' Preparer for text match.'''
import math
from absl import logging
from delta.data.preprocess.base_preparer import Preparer
from delta.data.preprocess.utils import prepare_embedding
from delta.data.preprocess.utils import prepare_vocab
from delta.data.preprocess.utils import prepare_vocab_from_config
from delta.data.preprocess.utils import get_pre_process_text_ds
from delta.data.preprocess.utils import run_two_sentence
from delta.utils.solver.solver_utils import get_session_conf
from delta.data import utils as data_utils

from delta import utils
from delta.utils.register import registers

# pylint: disable=too-many-instance-attributes


@registers.preparer.register
class TextMatchPreparer(Preparer):
  """Preparer for text match."""

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
    self.text_vocab_file_path = self.task_config['text_vocab']
    self.label_vocab_file_path = self.task_config['label_vocab']
    self.session_conf = get_session_conf(self.config)

  #pylint: disable=too-many-locals
  def _prepare_raw_data(self, pre_process_pipeline):
    """Preparing raw data."""
    all_texts = []
    all_labels = []
    for mode in self.all_modes:
      paths = self.config["data"][mode]['paths']
      paths_after_pre_process = [one_path + ".after" for one_path in paths]
      logging.debug(
          "paths_after_pre_process: {}".format(paths_after_pre_process))

      infer_without_label = bool(mode == utils.INFER and self.infer_no_label)

      for one_path, one_path_after in zip(paths, paths_after_pre_process):
        (text_left, text_right), label = data_utils.load_match_raw_data(
            paths=[one_path], mode=mode, infer_no_label=infer_without_label)
        text_t_left = get_pre_process_text_ds(text_left, pre_process_pipeline,
                                              self.num_parallel_calls, self.batch_size)
        text_t_right = get_pre_process_text_ds(text_right, pre_process_pipeline,
                                               self.num_parallel_calls, self.batch_size)

        batch_num = math.ceil(len(text_left) / float(self.batch_size))
        text_after_left_right = run_two_sentence([text_t_left, text_t_right], batch_num,
                                               self.session_conf)

        text_after_arr_l_r = [(left.decode("utf-8"), right.decode("utf-8")) for (left, right)
                              in text_after_left_right]
        all_texts += text_after_arr_l_r
        all_labels += label
        with open(one_path_after, "w", encoding="utf-8") as out_f:
          for i, [one_line_l, one_line_r] in enumerate(text_after_arr_l_r):
            if mode == utils.INFER and self.infer_no_label:
              out_f.write(one_line_l + '\t' + one_line_r + "\n")
            else:
              out_f.write(label[i] + "\t" + one_line_l + '\t' +
                          one_line_r + "\n")

    return all_texts, all_labels

  def _prepare_vocabs(self, all_texts, all_labels):
    """Preparing vocab for x."""
    logging.info("Preparing vocab for x ...")
    vocab_data = []
    for left_right in all_texts:
       vocab_data.extend(left_right)
    #
    prepare_vocab(
        self.text_vocab_file_path,
        vocab_data,
        min_frequency=self.vocab_min_frequency)

    logging.info("Preparing vocab for y ...")
    if "vocab" in self.config["data"]["task"]["classes"]:
      prepare_vocab_from_config(self.label_vocab_file_path, self.config)
    else:
      prepare_vocab(
          self.label_vocab_file_path,
          all_labels,
          min_frequency=1,
          use_default_dict=False)

  def _prepare_embedding(self):
    """Preparing embedding."""
    logging.info("Preparing embedding ...")
    if self.model_config["use_pre_train_emb"]:
      prepare_embedding(self.model_config["pre_train_emb_path"],
                        self.task_config["text_vocab"],
                        self.model_config["embedding_path"])

  def do_prepare(self, pre_process_pipeline):
    """Do the prepare processing."""

    all_texts, all_labels = self._prepare_raw_data(pre_process_pipeline)

    self._prepare_vocabs(all_texts, all_labels)

    self._prepare_embedding()
