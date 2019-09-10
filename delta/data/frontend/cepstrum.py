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

import tensorflow as tf

from delta.layers.ops import py_x_ops
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend

class Cepstrum(BaseFrontend):

  def __init__(self, config:dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    ''' set params '''

    window_length = 0.025
    frame_length = 0.010
    ceps_subband_num = 13
    tag_ceps_mean_norm = True

    hparams = HParams(cls=cls)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('ceps_subband_num', ceps_subband_num)
    hparams.add_hparam('tag_ceps_mean_norm', tag_ceps_mean_norm)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate):

    p = self.config

    with tf.name_scope('cepstrum'):

      cepstrum = py_x_ops.cepstrum(
        audio_data,
        sample_rate,
        window_length=p.window_length,
        frame_length=p.frame_length,
        ceps_subband_num=p.ceps_subband_num,
        tag_ceps_mean_norm=p.tag_ceps_mean_norm)

    return cepstrum
