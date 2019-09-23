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
import numpy as np
from scipy.io import wavfile

class ReadWav(BaseFrontend):

  def __init__(self, config:dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
      Set params.
       :param config: contains one optional parameters: sample_rate(float, default=16000.0).
       :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
       """
    sample_rate = 16000.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, wav_path):
    """
    Get audio data and sample rate from a wavfile.
    :param wav_path: filepath of wav
    :return: 2 values. The first is a Tensor of audio data. The second return value is the sample rate of the input wav
        file, which is a tensor with float dtype.
    """
    p = self.config

    sample_rate, pcm_data = wavfile.read(wav_path)
    assert float(sample_rate) == p.sample_rate, \
      "The wavfile's sample rate is not equal to the config's sample rate."

    audio_data = tf.constant(self.pcm2float(pcm_data), dtype=tf.float32)
    sample_rate = tf.constant(sample_rate, dtype=tf.float32)

    return audio_data, sample_rate

  def pcm2float(self, pcm_data):
    pcm_data = np.asarray(pcm_data)
    if pcm_data.dtype.kind not in 'iu':
      raise TypeError("'pcm_data' must be an array of integers.")
    i = np.iinfo(pcm_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (pcm_data.astype(dtype=np.float32) - offset) / abs_max
