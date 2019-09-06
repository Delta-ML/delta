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
''' Fbank '''

import abc
import numpy as np
import tensorflow as tf

from delta.data.frontend.base_frontend import BaseFrontend
from delta.utils.hparam import HParams

from delta.data.feat.tf_speech_feature import extract_feature

class FBank(BaseFrontend):

  def __init__(self, config:dict):
   super().__init__(config)

  @classmethod
  def params(cls, config=None):
    ''' set params '''

    sr = 16000
    bins = 40
    dither = True
    use_delta_deltas = True
    cmvn = False
    cmvn = ''

    if config is not None:
      taskconf = config['data']['task']
      audioconf = taskconf['audio']
      sr= taskconf['audio']['sr'],
      bins = audioconf['feature_size'],
      use_delta_deltas =     audioconf['add_delta_deltas'],
      cmvn = audioconf['cmvn'],
      cmvn_path = audioconf['cmvn_path']

    hparams = HParams(cls=cls)
    hparams.add_hparam("audio_sample_rate", sr)
    hparams.add_hparam("audio_channels", 1)
    hparams.add_hparam("audio_preemphasis", 0.97)
    if dither:
      hparams.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
    else:
      hparams.add_hparam("audio_dither", 0.0)
    hparams.add_hparam("audio_frame_length", 25.0)
    hparams.add_hparam("audio_frame_step", 10.0)
    hparams.add_hparam("audio_lower_edge_hertz", 20.0)
    hparams.add_hparam("audio_upper_edge_hertz", sr / 2.0)
    hparams.add_hparam("audio_num_mel_bins", bins)
    hparams.add_hparam("audio_add_delta_deltas", use_delta_deltas)
    hparams.add_hparam("num_zeropad_frames", 0)
    hparams.add_hparam("audio_global_cmvn", cmvn)

    return hparams

  def call(self, inputs):
    return extract_feature(inputs, params=self.config)
