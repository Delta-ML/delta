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
"""Task class for text classification."""

import collections
import delta.compat as tf
from absl import logging

from delta.data.task.base_text_task import TextTask

from delta.data.utils.common_utils import load_npy
from delta.data.utils.common_utils import get_file_len
from delta.data.preprocess.text_ops import process_one_label_dataset
from delta.data.preprocess.text_ops import load_dense_dataset
from delta.data.preprocess.utils import load_vocab_dict
from delta.data.preprocess.text_ops import load_textline_dataset
from delta import utils
from delta.utils.register import registers
from delta.layers.utils import compute_sen_lens

# pylint: disable=too-many-instance-attributes, too-many-locals


@registers.task.register
class TextClsTask(TextTask):
  """Task class for text classification."""

  def __init__(self, config, mode):
    super().__init__(config, mode)
    self.infer_no_label = self.config["data"][utils.INFER].get(
      'infer_no_label', False)
    self.vocab_min_frequency = self.task_config['vocab_min_frequency']
    self.text_vocab_file_path = self.task_config['text_vocab']
    self.label_vocab_file_path = self.task_config['label_vocab']
    self.max_seq_len = self.task_config['max_seq_len']
    self.num_classes = self.task_config['classes']['num_classes']
    self.use_true_length = self.model_config.get("use_true_length", False)
    self.split_token = self.model_config.get("split_token", "")
    self.use_dense = self.task_config["use_dense"]
    if self.use_dense:
      self.dense_input_dim = self.task_config["dense_input_dim"]
      self.dense_npy = config["data"][self.mode]["dense_npy"]
    self.paths = self.data_config[mode]['paths']
    self.paths_after_pre_process = [
        one_path + ".after" for one_path in self.paths
    ]
    self.infer_without_label = bool(mode == utils.INFER and self.infer_no_label)

    self.prepare()

  def generate_data(self):
    """Generate data for offline training."""
    if self.infer_without_label:
      column_num = 1
      text_ds = load_textline_dataset(self.paths_after_pre_process, column_num)
    else:
      column_num = 2
      label_ds, text_ds = load_textline_dataset(self.paths_after_pre_process, column_num)

    input_pipeline_func = self.get_input_pipeline(for_export=False)

    text_ds = text_ds.map(
        input_pipeline_func, num_parallel_calls=self.num_parallel_calls)

    text_size_ds = text_ds.map(
        lambda x: compute_sen_lens(x, padding_token=utils.PAD_IDX),
        num_parallel_calls=self.num_parallel_calls)

    text_ds = tf.data.Dataset.zip((text_ds, text_size_ds))

    if self.use_dense:
      dense = load_npy(self.dense_npy)
      dense_ds = load_dense_dataset(dense)

    if self.infer_without_label:
      if self.use_dense:
        data_set = tf.data.Dataset.zip((text_ds, dense_ds))
      else:
        data_set = text_ds
    else:
      label_ds = process_one_label_dataset(label_ds, self.config)
      if self.use_dense:
        data_set = tf.data.Dataset.zip((text_ds, dense_ds, label_ds))
      else:
        data_set = tf.data.Dataset.zip((text_ds, label_ds))

    vocab_dict = load_vocab_dict(self.text_vocab_file_path)
    vocab_size = len(vocab_dict)
    if self.use_true_length and self.split_token != "":
      if self.split_token not in vocab_dict:
        raise ValueError(
            "The Model uses split token: {}, not in corpus.".format(
                self.split_token))
      self.config['data']['split_token'] = int(vocab_dict[self.split_token])
    self.config['data']['vocab_size'] = vocab_size
    self.config['data']['{}_data_size'.format(self.mode)] = get_file_len(self.paths_after_pre_process)


    return data_set

  def feature_spec(self):
    """Get shapes for feature."""
    feature_shapes = [(tf.TensorShape([self.max_seq_len]), tf.TensorShape([]))]
    if self.use_dense:
      feature_shapes.append(tf.TensorShape(self.dense_input_dim))
    if not self.infer_without_label:
      feature_shapes.append(tf.TensorShape([self.num_classes]))
    if len(feature_shapes) == 1:
      return feature_shapes[0]
    return tuple(feature_shapes)

  def export_inputs(self):
    """Inputs for exported model."""
    vocab_dict = load_vocab_dict(self.text_vocab_file_path)
    vocab_size = len(vocab_dict)
    if self.use_true_length and self.split_token != "":
      if self.split_token not in vocab_dict:
        raise ValueError(
            "The Model uses split token: {}, not in corpus.".format(
                self.split_token))
      self.config['data']['split_token'] = int(vocab_dict[self.split_token])
    self.config['data']['vocab_size'] = vocab_size

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

    if self.use_dense:
      input_dense = tf.placeholder(
          shape=(None,), dtype=tf.float32, name="input_dense")
      export_data["export_inputs"]["input_dense"] = input_dense

    return export_data

  def dataset(self):
    """Data set function"""

    data_set = self.generate_data()
    logging.debug("data_set: {}".format(data_set))
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

    # pylint: disable=unused-variable
    if self.infer_without_label:
      if self.use_dense:
        (input_x, input_x_len), input_dense = iterator.get_next()
      else:
        input_x, input_x_len = iterator.get_next()
    else:
      if self.use_dense:
        (input_x, input_x_len), input_dense, input_y = iterator.get_next()
      else:
        (input_x, input_x_len), input_y = iterator.get_next()

    input_x_dict = collections.OrderedDict([("input_x", input_x)])
    return_dict = {
        "input_x_dict": input_x_dict,
        "input_x_len": input_x_len,
        "iterator": iterator,
    }

    if self.use_dense:
      input_x_dict["input_dense"] = input_dense

    if not self.infer_without_label:
      return_dict["input_y_dict"] = collections.OrderedDict([("input_y",
                                                              input_y)])

    return return_dict
