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
''' sequence labeling task '''

import os
import collections
import tensorflow as tf
from absl import logging

from delta.data.task.base_text_task import TextTask
from delta.data.utils.common_utils import load_seq_label_raw_data
from delta.data.utils.common_utils import load_multi_label_dataset
from delta.data.preprocess.utils import get_vocab_size
from delta.data.preprocess.text_ops import tokenize_sentence
from delta.data.preprocess.text_ops import pre_process_text
from delta import utils
from delta.utils.register import registers
from delta.layers.utils import compute_sen_lens

# pylint: disable=too-many-instance-attributes


@registers.task.register
class TextSeqLabelTask(TextTask):
  """Task class for sequence labeling."""

  def __init__(self, config, mode):
    super().__init__(config, mode)
    data_config = config['data']

    self.mode = mode
    self.paths = data_config[mode]['paths']
    self.infer_no_label = data_config[utils.INFER].get('infer_no_label', False)
    if self.mode == utils.INFER and self.infer_no_label:
      self.infer_without_label = True
    else:
      self.infer_without_label = False
    self.paths_after_pre_process = []

    task_config = data_config['task']
    self.vocab_min_frequency = task_config['vocab_min_frequency']
    self.text_vocab_file_path = task_config['text_vocab']
    self.label_vocab_file_path = task_config['label_vocab']
    self.max_seq_len = task_config['max_seq_len']
    self.num_classes = task_config["classes"]['num_classes']
    self.batch_size = task_config['batch_size']
    self.epochs = task_config['epochs']
    self.num_parallel_calls = task_config['num_parallel_calls']
    self.num_prefetch_batch = task_config['num_prefetch_batch']
    self.shuffle_buffer_size = task_config['shuffle_buffer_size']
    self.need_shuffle = task_config['need_shuffle']
    self.task_config = task_config
    self.model_config = config["model"]
    self.language = self.task_config["language"]
    self.split_by_space = self.task_config.get("split_by_space", False)
    self.use_word = self.task_config.get("use_word", False)
    self.paths_after_pre_process = [
        one_path + ".after" for one_path in self.paths
    ]
    self.init_feed_dict = {}
    self.prepare()

  def pre_process_pipeline(self, input_sentences):
    """Data pipeline function for pre-processing."""
    batch = pre_process_text(input_sentences, self.language,
                             self.split_by_space, self.use_word)
    return batch

  def common_process_pipeline(self, batch):
    """
    Data pipeline function for common process.
    This function is used both by online training and offline inference.
    """
    vocab_path = os.path.abspath(self.text_vocab_file_path)
    token_ids = tokenize_sentence(batch, self.max_seq_len, vocab_path)
    return token_ids

  def load_text_dataset(self, text_placeholder):
    """Load text data set."""
    logging.info("Loading text dataset...")
    text_ds = tf.data.Dataset.from_tensor_slices(text_placeholder)
    input_pipeline_func = self.get_input_pipeline(for_export=False)
    text_ds = text_ds.map(
        input_pipeline_func, num_parallel_calls=self.num_parallel_calls)
    text_size_ds = text_ds.map(
        lambda x: compute_sen_lens(x, padding_token=0),
        num_parallel_calls=self.num_parallel_calls)
    text_ds = tf.data.Dataset.zip((text_ds, text_size_ds))

    return text_ds

  def generate_data(self):
    """Generate data for offline training."""
    text, label = load_seq_label_raw_data(
        paths=self.paths, mode=self.mode, infer_no_label=self.infer_no_label)

    text_placeholder = tf.placeholder(tf.string, name="text")
    label_placeholder = tf.placeholder(tf.string, name="label")
    self.init_feed_dict[text_placeholder] = text
    self.init_feed_dict[label_placeholder] = label

    text_ds = self.load_text_dataset(text_placeholder)

    if self.infer_without_label:
      data_set = text_ds
    else:
      label_ds = load_multi_label_dataset(label_placeholder, self.config)
      data_set = tf.data.Dataset.zip((text_ds, label_ds))

    self.config['data']['vocab_size'] = get_vocab_size(
        self.text_vocab_file_path)
    self.config['data']['{}_data_size'.format(self.mode)] = len(text)

    return data_set

  def feature_spec(self):
    """Get shapes for feature."""
    feature_shapes = [(tf.TensorShape([self.max_seq_len]), tf.TensorShape([]))]
    if not self.infer_without_label:
      feature_shapes.append(tf.TensorShape([self.max_seq_len]))
    if len(feature_shapes) == 1:
      return feature_shapes[0]
    return tuple(feature_shapes)

  def export_inputs(self):
    """Inputs for exported model."""
    self.config['data']['vocab_size'] = get_vocab_size(
        self.text_vocab_file_path)
    input_sentence = tf.placeholder(
        shape=(None,), dtype=tf.string, name="input_sentence")

    input_pipeline_func = self.get_input_pipeline(for_export=True)
    token_ids = input_pipeline_func(input_sentence)
    token_ids_len = tf.map_fn(lambda x: compute_sen_lens(x, padding_token=0),
                              token_ids)

    export_data = {
        "export_inputs": {
            "input_sentence": input_sentence
        },
        "model_inputs": {
            "input_x": token_ids,
            "input_x_len": token_ids_len
        }
    }

    return export_data

  def dataset(self):
    """Dataset function"""
    data_set = self.generate_data()

    if self.mode == 'train':
      if self.need_shuffle:
        # shuffle batch size and repeat
        logging.debug("shuffle and repeat dataset ...")
        data_set = data_set.apply(
          tf.data.experimental.shuffle_and_repeat(
            buffer_size=self.shuffle_buffer_size, count=None))
      else:
        logging.debug("repeat dataset ...")
        data_set = data_set.repeat(count=None)

    feature_shape = self.feature_spec()
    logging.debug("feature_shape: {}".format(feature_shape))

    data_set = data_set.padded_batch(
        batch_size=self.batch_size, padded_shapes=feature_shape)

    data_set = data_set.prefetch(self.num_prefetch_batch)

    iterator = data_set.make_initializable_iterator()

    if self.infer_without_label:
      input_x, input_x_len = iterator.get_next()
    else:
      (input_x, input_x_len), input_y = iterator.get_next()

    input_x_dict = collections.OrderedDict([("input_x", input_x)])
    return_dict = {
        "input_x_dict": input_x_dict,
        "input_x_len": input_x_len,
        "iterator": iterator,
        "init_feed_dict": self.init_feed_dict
    }

    if not self.infer_without_label:
      return_dict["input_y_dict"] = collections.OrderedDict([("input_y",
                                                              input_y)])

    return return_dict
