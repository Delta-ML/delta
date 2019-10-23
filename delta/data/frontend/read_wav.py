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

from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend


class ReadWav(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
      Set params.
       :param config: contains two optional parameters: audio_channels(int, default=1),
              sample_rate(float, default=16000.0).
       :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
       """
    audio_channels = 1
    sample_rate = 16000.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('audio_channels', audio_channels)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, wavfile):
    """
    Get audio data and sample rate from a wavfile.
    :param wavfile: filepath of wav
    :return: 2 values. The first is a Tensor of audio data. The second return value is the sample rate of the input wav
        file, which is a tensor with float dtype.
    """
    p = self.config
    contents = tf.io.read_file(wavfile)
    audio_data, sample_rate = tf.audio.decode_wav(
        contents, desired_channels=p.audio_channels)
    assert_op = tf.assert_equal(
        tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=float))
    with tf.control_dependencies([assert_op]):
      return tf.squeeze(audio_data, axis=-1), tf.cast(sample_rate, dtype=float)
