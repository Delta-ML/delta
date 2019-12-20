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
"""This model extracts zcr features per frame."""

import delta.compat as tf

from core.ops import py_x_ops
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend


class Zcr(BaseFrontend):
  """
  Compute ZCR features respectively，and concate them. Return
  a tensor with shape (1, num_frames).
  """

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config:contains three optional parameters:
        --sample_rate       : Waveform data sample frequency (must match the waveform
                             file, if specified there). (float, default = 16000)
        --window_length		 : Window length in seconds. (float, default = 0.025)
        --frame_length		 : Hop length in seconds. (float, default = 0.010)
    :return: An object of class HParams, which is a set of hyperparameters as
            name-value pairs.
    """

    window_length = 0.025
    frame_length = 0.010
    sample_rate = 16000

    hparams = HParams(cls=cls)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
    Calculate the zero-crossing rate of speech.
    :param audio_data: the audio signal from which to compute spectrum.
                      Should be an (1, N) tensor.
    :param sample_rate: [option]the samplerate of the signal we working with,
                        default is 16kHz.
    :return: A tensor with shape (1, num_frames), containing zero-crossing rate of
            every frame in speech.
    """

    p = self.config
    with tf.name_scope('zcr'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))
      with tf.control_dependencies([assert_op]):

        sample_rate = tf.cast(sample_rate, dtype=float)
        zcr = py_x_ops.zcr(
            audio_data,
            sample_rate,
            window_length=p.window_length,
            frame_length=p.frame_length)

        return zcr
