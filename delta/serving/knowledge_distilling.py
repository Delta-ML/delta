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
'''Teacher module'''
import numpy as np
import delta.compat as tf

from delta.serving.base_frozen_model import FrozenModel
from delta.utils.register import registers


@registers.serving.register
class Teacher(FrozenModel):
  '''class of Teacher'''

  def __init__(self, model, gpu_str=None, temperature=1.0):
    '''
     model: saved model dir, ckpt dir or frozen_graph_pb path
     gpu_str: list of gpu devices. e.g. '' for cpu, '0,1' for gpu 0,1
    '''
    super().__init__(model, gpu_str)

    self.temperature = temperature
    self.build()

  def build(self):
    '''build'''
    self.audio_ph = self.graph.get_tensor_by_name('inputs:0')
    self.logits = self.graph.get_tensor_by_name(
        'model/logits/logits-matmul/add:0')

    with self.graph.as_default():
      self.soft_label = tf.nn.softmax(self.logits / self.temperature)

  def __call__(self, feat):
    ''' generate soft labels per example '''
    # shape [1, T, D, C]
    inputs = feat[np.newaxis, :, :, :]
    validate_feed = {
        self.audio_ph: inputs,
    }
    soft_label = self.sess.run(self.soft_label, feed_dict=validate_feed)
    return soft_label[0]
