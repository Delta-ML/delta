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
"""Base class for text classification."""

import delta.compat as tf

from delta.models.base_model import Model
from delta.layers.utils import get_pad_mask_from_token_idx
from delta.layers.utils import get_seg_mask_from_token_idx


class TextClassModel(Model):  # pylint: disable=abstract-method
  """Base class for text classification."""

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    model_config = config['model']['net']['structure']
    if "dense_input" in model_config:
      self.use_dense_task = True
      self.use_dense_input = model_config["dense_input"]["use_dense_input"]
      self.only_dense_input = model_config["dense_input"]["only_dense_input"]
      self.dense_input_dim = config['data']['task']['dense_input_dim']
    else:
      self.use_dense_task = False
      self.use_dense_input = False

    if self.use_dense_input:
      self.dense_input_linear = tf.keras.layers.Dense(
          self.dense_input_dim, activation=tf.keras.activations.linear)

    self.use_pretrained_model = config['model'].get('use_pre_train_model',
                                                    False)
    if self.use_pretrained_model:
      pretrained_model_config = config['model']['pre_train_model']
      self.pretrained_model_name = pretrained_model_config['name']
      self.pretrained_model_mode = pretrained_model_config['mode']
      self.pretrained_model_dim = pretrained_model_config['dim']
      self.pretrained_model_path = pretrained_model_config['path']
      self.pretrained_model_seg = pretrained_model_config['seg']
      self.pretrained_model_cls = pretrained_model_config['cls']
      self.pretrained_model_pad = pretrained_model_config['pad']
      self.pretrained_model_layers = pretrained_model_config['layers']
      self.pretrained_model_output = pretrained_model_config['output']

  def get_pre_train_graph(self, inputs):
    pretrained_model_meta = self.pretrained_model_path + '.meta'
    seg_idx = self.pretrained_model_seg
    pad_idx = self.pretrained_model_pad
    with tf.name_scope('pretrain_graph') as scope:
      pretrained_graph = tf.get_default_graph()
      if self.pretrained_model_name == 'elmo':
        pretrained_saver = tf.train.import_meta_graph(
            pretrained_model_meta, input_map={'input_x:0': inputs})
        input_x_pretrained = pretrained_graph.get_tensor_by_name(
            scope + 'input_x_elmo:0')
      if self.pretrained_model_name == 'bert':
        pad_mask = get_pad_mask_from_token_idx(inputs, pad_idx)
        segment_mask = get_seg_mask_from_token_idx(inputs, seg_idx)
        pretrained_saver = tf.train.import_meta_graph(
            pretrained_model_meta,
            input_map={
                'input_ids:0': inputs,
                'input_mask:0': pad_mask,
                'segment_ids:0': segment_mask
            })
        if self.pretrained_model_output == 'seq':
          input_x_pretrained = \
            pretrained_graph.get_tensor_by_name(scope + 'encoder_layers_{}:0'.
                                                format(self.pretrained_model_layers))
        else:
          input_x_pretrained = pretrained_graph.get_tensor_by_name(
              scope + 'input_x_bert_cls:0')
      return input_x_pretrained
