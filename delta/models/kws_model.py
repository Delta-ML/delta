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
''' tdnn kws model '''
import delta.compat as tf

from delta.models.base_model import RawModel

from delta.utils.register import registers


@registers.model.register
class TdnnKwsModel(RawModel):
  ''' main model '''

  def __init__(self, config, name=None):
    super().__init__(name=name)
    self.cfg = config
    self.train = None

  #pylint: disable=arguments-differ
  def call(self, features, **kwargs):
    self.train = kwargs['training']
    n_class = self.cfg['data']['task']['classes']['num']
    return self.tdnn(features, n_class, self.train)

  def tdnn(self, features, n_class, is_train):
    '''
        inp: (batch_size, window_len, feat_dim)
    '''
    inp = features['inputs']
    kernel_size = self.cfg['model']['net']['kernel_size']
    strides = self.cfg['model']['net']['strides']
    num_layers = self.cfg['model']['net']['num_layers']
    filters_num = inp.get_shape()[-1]

    for i in range(num_layers):
      output = tf.nn.relu(
          tf.layers.conv1d(inp, filters_num, kernel_size, strides=strides))
      output = tf.layers.batch_normalization(
          output, training=is_train, name='bn%d' % i)
      inp = output

    dim = output.get_shape()[1] * output.get_shape()[2]
    output = tf.reshape(output, [-1, dim])

    logits = tf.layers.dense(output, n_class)
    return logits
