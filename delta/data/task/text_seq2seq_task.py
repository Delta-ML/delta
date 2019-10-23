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
"""Task class for text sequence to sequence."""

import collections
import os

import delta.compat as tf
from absl import logging

from delta import utils
from delta.data.preprocess.text_ops import tokenize_sentence
from delta.data.preprocess.utils import load_vocab_dict
from delta.data.task.base_text_task import TextTask
from delta.data.preprocess.text_ops import load_textline_dataset
from delta.data.utils.common_utils import get_file_len
from delta.layers.utils import compute_sen_lens
from delta.utils.register import registers

# pylint: disable=too-many-instance-attributes, too-many-locals


@registers.task.register
class TextS2STask(TextTask):
  """Task class for text sequence to sequence."""
  START_TOKEN = '<sos>'
  END_TOKEN = '<eos>'

  def __init__(self, config, mode):
    super().__init__(config, mode)

    self.src_paths = self.data_config[mode]['paths']['source']
    self.tgt_paths = self.data_config[mode]['paths']['target']

    self.vocab_min_frequency = self.task_config['vocab_min_frequency']
    self.text_vocab_file_path = self.task_config['text_vocab']
    self.label_vocab_file_paths = self.task_config['label_vocab']
    if not isinstance(self.label_vocab_file_paths, list):
      self.label_vocab_file_paths = [self.label_vocab_file_paths]
    self.use_label_vocab = self.task_config['use_label_vocab']

    self.max_enc_len = self.task_config['max_enc_len']
    self.max_dec_len = self.task_config['max_dec_len']

    self.src_paths_after_pre_process = [
        one_path + ".after" for one_path in self.src_paths
    ]
    self.tgt_paths_after_pre_process = [
        one_path + ".after" for one_path in self.tgt_paths
    ]
    self.infer_no_label = self.config["data"][utils.INFER].get(
      'infer_no_label', False)
    self.infer_without_label = bool(mode == utils.INFER and self.infer_no_label)

    self.prepare()

  def common_process_pipeline(self, batch):
    """
    Data pipeline function for common process.
    This function is used both by online training and offline inference.
    """
    vocab_path = os.path.abspath(self.text_vocab_file_path)
    token_ids = self.text_pipeline_func(batch, self.max_enc_len, vocab_path)
    return token_ids

  def text_pipeline_func(self, batch, seq_len, vocab_path):
    """
    Data pipeline function for core text process.
    """
    vocab_path = os.path.abspath(vocab_path)
    token_ids = tokenize_sentence(batch, seq_len, vocab_path)
    return token_ids

  def exclude_padding(self, batch):
    x_binary = tf.cast(tf.not_equal(batch, utils.PAD_IDX), tf.int32)
    sen_lens = tf.reduce_sum(x_binary, axis=-1)
    return batch[:sen_lens]

  def generate_data(self):
    """Generate data for offline training."""

    column_num = 1
    src_path = self.src_paths_after_pre_process
    target_path = self.tgt_paths_after_pre_process

    src_ds = load_textline_dataset([src_path], column_num)

    src_ds = src_ds[0]

    input_pipeline_func = self.get_input_pipeline(for_export=False)

    src_ds = src_ds.map(
      input_pipeline_func, num_parallel_calls=self.num_parallel_calls)

    src_size_ds = src_ds.map(
      lambda x: compute_sen_lens(x, padding_token=utils.PAD_IDX),
      num_parallel_calls=self.num_parallel_calls)

    src_ds = src_ds.map(
      self.exclude_padding, num_parallel_calls=self.num_parallel_calls)

    if self.infer_without_label:
      data_set = tf.data.Dataset.zip((src_ds, src_size_ds))

    else:
      tgt = load_textline_dataset([target_path], column_num)
      tgt = tgt[0]
      tgt_out_ds = tgt.map(lambda x: x + ' ' + self.END_TOKEN)
      tgt_in_ds = tgt.map(lambda x: self.START_TOKEN + ' ' + x)

      tgt_in_ds = tgt_in_ds.map(
        lambda batch: self.text_pipeline_func(batch, self.max_dec_len, self.
                                              text_vocab_file_path),
        num_parallel_calls=self.num_parallel_calls)

      tgt_in_size_ds = tgt_in_ds.map(
        lambda x: compute_sen_lens(x, padding_token=utils.PAD_IDX),
        num_parallel_calls=self.num_parallel_calls)

      tgt_in_ds = tgt_in_ds.map(
        self.exclude_padding, num_parallel_calls=self.num_parallel_calls)

      inp_ds = tf.data.Dataset.zip(
        (src_ds, src_size_ds, tgt_in_ds, tgt_in_size_ds))

      if self.use_label_vocab:
        target_vocab_file_path = self.label_vocab_file_paths[0]
      else:
        target_vocab_file_path = self.text_vocab_file_path
      tgt_out_ds = tgt_out_ds.map(
          lambda batch: self.text_pipeline_func(batch, self.max_dec_len,
                                                target_vocab_file_path),
          num_parallel_calls=self.num_parallel_calls)

      tgt_out_ds = tgt_out_ds.map(
          self.exclude_padding, num_parallel_calls=self.num_parallel_calls)
      data_set = tf.data.Dataset.zip((inp_ds, tgt_out_ds))

    vocab_dict = load_vocab_dict(self.text_vocab_file_path)
    vocab_size = len(vocab_dict)
    label_vocab_dict = load_vocab_dict(self.label_vocab_file_paths[0])
    label_vocab_size = len(label_vocab_dict)
    data_size = get_file_len(self.src_paths_after_pre_process)
    self.config['data']['vocab_size'] = vocab_size
    self.config['data']['label_vocab_size'] = label_vocab_size
    self.config['data']['{}_data_size'.format(self.mode)] = data_size

    return data_set

  def feature_spec(self):
    """Get shapes for feature."""
    if not self.infer_without_label:
      feature_shapes = [(tf.TensorShape([tf.Dimension(None)]), tf.TensorShape([]),
                         tf.TensorShape([tf.Dimension(None)]), tf.TensorShape([]))]
      feature_shapes.append(tf.TensorShape([tf.Dimension(None)]))
    else:
      feature_shapes = [(tf.TensorShape([tf.Dimension(None)]), tf.TensorShape([]))
                        ]
    if len(feature_shapes) == 1:
      return feature_shapes[0]
    return tuple(feature_shapes)

  def export_inputs(self):
    """Inputs for exported model."""
    vocab_dict = load_vocab_dict(self.text_vocab_file_path)
    vocab_size = len(vocab_dict)
    label_vocab_dict = load_vocab_dict(self.label_vocab_file_paths[0])
    label_vocab_size = len(label_vocab_dict)
    self.config['data']['vocab_size'] = vocab_size
    self.config['data']['label_vocab_size'] = label_vocab_size

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
            "input_enc_x": token_ids,
            "input_x_len": token_ids_len
        }
    }

    return export_data

  def dataset(self):
    """Data set function"""

    data_set = self.generate_data()
    logging.debug("data_set: {}".format(data_set))
    if self.need_shuffle and self.mode == 'train':
      # shuffle batch size and repeat
      logging.debug("shuffle dataset ...")
      data_set = data_set.apply(
          tf.data.experimental.shuffle_and_repeat(
              buffer_size=self.shuffle_buffer_size, count=None))

    feature_shape = self.feature_spec()
    logging.debug("feature_shape: {}".format(feature_shape))
    data_set = data_set.padded_batch(
        batch_size=self.batch_size, padded_shapes=feature_shape)

    data_set = data_set.prefetch(self.num_prefetch_batch)

    iterator = data_set.make_initializable_iterator()

    # pylint: disable=unused-variable
    if self.infer_without_label:
      input_enc_x, input_enc_x_len = iterator.get_next()
      input_x_dict = collections.OrderedDict([("input_enc_x", input_enc_x)])

    else:
      (input_enc_x, input_enc_x_len, input_dec_x,
       input_dec_x_len), input_y = iterator.get_next()

      input_x_dict = collections.OrderedDict([("input_enc_x", input_enc_x),
                                              ("input_dec_x", input_dec_x)])

    return_dict = {
        "input_x_dict": input_x_dict,
        "input_x_len": input_enc_x_len,
        "iterator": iterator,
    }

    if not self.infer_without_label:
      return_dict["input_y_dict"] = collections.OrderedDict([("input_y",
                                                              input_y)])
      return_dict["input_y_len"] = input_dec_x_len

    return return_dict
