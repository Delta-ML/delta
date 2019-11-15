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

import delta.compat as tf

from delta.layers.ops import py_x_ops
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend


class DeltaDelta(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):

    hparams = HParams(cls=cls)

    return hparams

  def call(self, feat, order, window):
    """
    Caculate delta of feats.
    :param feat: a float tensor of size (num_frames, dim_feat).
    :param order: an int.
    :param window: an int.
    :return: A tensor with shape (num_frames, (dim_feat * (order + 1))),
        containing delta of features of every frame in speech.
    """

    p = self.config
    with tf.name_scope('delta_delta'):
      delta_delta = py_x_ops.delta_delta(feat, order, window)

    return delta_delta
