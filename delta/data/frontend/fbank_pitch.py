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

from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend
from delta.data.frontend.pitch import Pitch
from delta.data.frontend.fbank import Fbank

class Fbank_pitch(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)
    self.fbank = Fbank(config)
    self.pitch = Pitch(config)

  @classmethod
  def params(cls, config=None):
    ''' set params '''

    upper_frequency_limit = 4000
    lower_frequency_limit = 20
    filterbank_channel_count = 40
    window_length = 0.025
    frame_length = 0.010
    thres_autoc = 0.3
    output_type = 2
    sample_rate = 16000

    hparams = HParams(cls=cls)
    hparams.add_hparam('upper_frequency_limit', upper_frequency_limit)
    hparams.add_hparam('lower_frequency_limit', lower_frequency_limit)
    hparams.add_hparam('filterbank_channel_count', filterbank_channel_count)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('thres_autoc', thres_autoc)
    hparams.add_hparam('output_type', output_type)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):

    p = self.config
    if sample_rate == None:
      sample_rate = tf.constant(p.sample_rate, dtype=tf.float32)

    with tf.name_scope('fbank_pitch'):

      fbank_feats = tf.squeeze(self.fbank(audio_data, sample_rate))
      pitch_feats = self.pitch(audio_data, sample_rate)
      # return shape = (num_frames * num_features)
      fbank_pitch_feats = tf.concat([fbank_feats, pitch_feats], 1)

    return fbank_pitch_feats
