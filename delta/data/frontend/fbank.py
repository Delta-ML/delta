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
from delta.data.frontend.spectrum import Spectrum

class Fbank(BaseFrontend):

  def __init__(self, config:dict):
    super().__init__(config)
    self.spect = Spectrum(config)


  @classmethod
  def params(cls, config=None):
    ''' set params '''

    upper_frequency_limit = 4000
    lower_frequency_limit = 20
    filterbank_channel_count = 40
    window_length = 0.025
    frame_length = 0.010
    output_type = 2

    hparams = HParams(cls=cls)
    hparams.add_hparam('upper_frequency_limit', upper_frequency_limit)
    hparams.add_hparam('lower_frequency_limit', lower_frequency_limit)
    hparams.add_hparam('filterbank_channel_count', filterbank_channel_count)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('output_type', output_type)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate):

    with tf.name_scope('fbank'):

      p = self.config
      spectrum = self.spect(audio_data, sample_rate)

      spectrum = tf.expand_dims(spectrum, 0)
      sample_rate = tf.to_int32(sample_rate)

      fbank = py_x_ops.fbank(
        spectrum,
        sample_rate,
        upper_frequency_limit=p.upper_frequency_limit,
        lower_frequency_limit=p.lower_frequency_limit,
        filterbank_channel_count=p.filterbank_channel_count)

    return fbank
