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
"""Task class for text match."""

import collections
from absl import logging
import delta.compat as tf

from delta.data.task.base_text_task import TextTask
from delta.data.utils.common_utils import get_file_len
from delta.data.preprocess.text_ops import load_textline_dataset
from delta.data.preprocess.utils import load_vocab_dict
from delta.data.preprocess.text_ops import process_one_label_dataset
from delta.utils.register import registers
from delta.layers.utils import compute_sen_lens
from delta import utils
# pylint: disable=too-many-instance-attributes


@registers.task.register
class TextMatchTask(TextTask):
  """Task class for text Match."""

  def __init__(self, config, mode):
    super().__init__(config, mode)

    self.vocab_min_frequency = self.task_config['vocab_min_frequency']
    self.text_vocab_file_path = self.task_config['text_vocab']
    self.label_vocab_file_path = self.task_config['label_vocab']
    self.max_seq_len = self.task_config['max_seq_len']
    self.num_classes = self.task_config['classes']['num_classes']

    self.paths = self.data_config[mode]['paths']
    self.paths_after_pre_process = [
        one_path + ".after" for one_path in self.paths
    ]
    self.infer_no_label = self.config["data"][utils.INFER].get(
      'infer_no_label', False)
    self.infer_without_label = bool(mode == utils.INFER and self.infer_no_label)

    self.prepare()

  # pylint: disable=too-many-locals
  def generate_data(self):
    """Generate data for offline training."""
    if self.infer_without_label:
      column_num=2
      text_ds_left, text_ds_right = load_textline_dataset(self.paths_after_pre_process, column_num)
    else:
      column_num=3
      label,text_ds_left, text_ds_right=load_textline_dataset(self.paths_after_pre_process, column_num)

    input_pipeline_func = self.get_input_pipeline(for_export=False)
    text_ds_left = text_ds_left.map(
        input_pipeline_func, num_parallel_calls=self.num_parallel_calls)
    text_ds_right = text_ds_right.map(
        input_pipeline_func, num_parallel_calls=self.num_parallel_calls)
    text_size_ds_left = text_ds_left.map(
        lambda x: compute_sen_lens(x, padding_token=0),
        num_parallel_calls=self.num_parallel_calls)
    text_size_ds_right = text_ds_right.map(
        lambda x: compute_sen_lens(x, padding_token=0),
        num_parallel_calls=self.num_parallel_calls)
    text_ds_left_right = tf.data.Dataset.zip((text_ds_left, text_ds_right))
    text_len_left_right = tf.data.Dataset.zip(
        (text_size_ds_left, text_size_ds_right))
    if self.infer_without_label:
      data_set_left_right = text_ds_left_right
    else:
      label_ds = process_one_label_dataset(label, self.config)
      data_set_left_right = tf.data.Dataset.zip((text_ds_left_right, label_ds))
    vocab_dict = load_vocab_dict(self.text_vocab_file_path)
    vocab_size = len(vocab_dict)

    self.config['data']['vocab_size'] = vocab_size
    self.config['data']['{}_data_size'.format(self.mode)] = get_file_len(self.paths_after_pre_process)

    return data_set_left_right, text_len_left_right

  def feature_spec(self):
    """Get shapes for feature."""
    feature_shapes = [(tf.TensorShape([self.max_seq_len]),
                       tf.TensorShape([self.max_seq_len]))]
    if not self.infer_without_label:
      feature_shapes.append(tf.TensorShape([self.num_classes]))
    if len(feature_shapes) == 1:
      return feature_shapes[0]
    return tuple(feature_shapes)

  def export_inputs(self):
    """Inputs for exported model."""
    vocab_dict = load_vocab_dict(self.text_vocab_file_path)
    vocab_size = len(vocab_dict)
    self.config['data']['vocab_size'] = vocab_size

    input_sent_left = tf.placeholder(
        shape=(None,), dtype=tf.string, name="input_sent_left")
    input_sent_right = tf.placeholder(
        shape=(None,), dtype=tf.string, name="input_sent_right")
    input_pipeline_func = self.get_input_pipeline(for_export=True)

    token_ids_left = input_pipeline_func(input_sent_left)
    token_ids_right = input_pipeline_func(input_sent_right)
    token_ids_len_left = tf.map_fn(
        lambda x: compute_sen_lens(x, padding_token=0), token_ids_left)
    token_ids_len_right = tf.map_fn(
        lambda x: compute_sen_lens(x, padding_token=0), token_ids_right)
    export_data = {
        "export_inputs": {
            "input_sent_left": input_sent_left,
            "input_sent_right": input_sent_right,
        },
        "model_inputs": {
            "input_x_left": token_ids_left,
            "input_x_right": token_ids_right,
            "input_x_len": [token_ids_len_left, token_ids_len_right]
        }
    }
    return export_data

  def dataset(self):
    """Data set function"""
    data_set_left_right, text_len_left_right = self.generate_data()

    logging.debug("data_set_left_right: {}".format(data_set_left_right))
    if self.mode == 'train':
      if self.need_shuffle:
        # shuffle batch size and repeat
        logging.debug("shuffle and repeat dataset ...")
        data_set_left_right = data_set_left_right.apply(
            tf.data.experimental.shuffle_and_repeat(
                buffer_size=self.shuffle_buffer_size, count=None))
      else:
        logging.debug("repeat dataset ...")
        data_set_left_right = data_set_left_right.repeat(count=None)
    feature_shape = self.feature_spec()
    logging.debug("feature_shape: {}".format(feature_shape))

    data_set_left_right = data_set_left_right.padded_batch(
        batch_size=self.batch_size, padded_shapes=feature_shape)
    text_len_left_right = text_len_left_right.batch(self.batch_size)

    data_set_left_right = data_set_left_right.prefetch(self.num_prefetch_batch)
    text_len_left_right = text_len_left_right.prefetch(self.num_prefetch_batch)

    iterator = data_set_left_right.make_initializable_iterator()
    iterator_len = text_len_left_right.make_initializable_iterator()
    # pylint: disable=unused-variable
    if self.infer_without_label:
      input_x_left, input_x_right = iterator.get_next()
    else:

      (input_x_left, input_x_right), input_y = iterator.get_next()

    input_x_left_len, input_x_right_len = iterator_len.get_next()
    input_x_dict = collections.OrderedDict([("input_x_left", input_x_left),
                                            ("input_x_right", input_x_right)])
    input_x_len = collections.OrderedDict([
        ("input_x_left_len", input_x_left_len),
        ("input_x_right_len", input_x_right_len)
    ])
    return_dict = {
        "input_x_dict": input_x_dict,
        "input_x_len": input_x_len,
        "iterator": iterator,
        "iterator_len": iterator_len,
    }

    if not self.infer_without_label:
      return_dict["input_y_dict"] = collections.OrderedDict([("input_y",
                                                              input_y)])
    return return_dict
