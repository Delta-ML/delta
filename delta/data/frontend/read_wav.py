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
"""The model reads audio sample from wav file."""

import delta.compat as tf
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend
from core.ops import py_x_ops


class ReadWav(BaseFrontend):
  """
      Read audio sample from wav file, return sample data and sample rate.
      """

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains three optional parameters:
          --sample_rate       : Waveform data sample frequency (must match the waveform
                                file, if specified there). (float, default = 16000)
          --speed             : Speed of sample channels wanted. (float, default=1.0)
          --audio_channels    :(int, default=1).
    :return: An object of class HParams, which is a set of hyperparameters as
            name-value pairs.
    """
    audio_channels = 1
    sample_rate = 16000
    speed = 1.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('audio_channels', audio_channels)
    hparams.add_hparam('sample_rate', sample_rate)
    hparams.add_hparam('speed', speed)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, wavfile):
    """
    Get audio data and sample rate from a wavfile.
    :param wavfile: filepath of wav.
    :return: 2 values. The first is a Tensor of audio data.
        The second return value isthe sample rate of the input wav
        file, which is a tensor with float dtype.
    """
    p = self.config
    contents = tf.io.read_file(wavfile)
    audio_data, sample_rate = tf.audio.decode_wav(
        contents, desired_channels=p.audio_channels)
    assert_op = tf.assert_equal(
        tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))

    with tf.control_dependencies([assert_op]):

      if p.speed == 1.0:
        return tf.squeeze(
            audio_data * 32768, axis=-1), tf.cast(
                sample_rate, dtype=tf.int32)
      else:
        resample_rate = tf.cast(
            sample_rate, dtype=tf.float32) * tf.cast(
                1.0 / p.speed, dtype=tf.float32)
        speed_data = py_x_ops.speed(
            tf.squeeze(audio_data * 32768, axis=-1),
            tf.cast(sample_rate, dtype=tf.int32),
            tf.cast(resample_rate, dtype=tf.int32),
            lowpass_filter_width=5)
        return tf.squeeze(speed_data), tf.cast(sample_rate, dtype=tf.int32)
