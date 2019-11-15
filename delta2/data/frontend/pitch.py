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


class Pitch(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: config: contains four optional parameters:window_length(float, default=0.025),
          frame_length(float, default=0.010), thres_autoc(float, default=0.3), sample_rate(float, default=16000.0).
    :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
    """

    window_length = 0.025
    frame_length = 0.010
    thres_autoc = 0.3
    sample_rate = 16000.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('thres_autoc', thres_autoc)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
    Caculate pitch features of audio data.
    :param audio_data: the audio signal from which to compute spectrum. Should be an (1, N) tensor.
    :param sample_rate: [option]the samplerate of the signal we working with, default is 16kHz.
    :return: A float tensor of size (1, num_frames) containing pitch features of every frame in speech.
    """

    p = self.config
    with tf.name_scope('pitch'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=float)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=float))
      with tf.control_dependencies([assert_op]):

        pitch = py_x_ops.pitch(
            audio_data,
            sample_rate,
            window_length=p.window_length,
            frame_length=p.frame_length,
            thres_autoc=p.thres_autoc)

        pitch = tf.squeeze(pitch)
        pitch = tf.transpose(pitch[None, :])
        return pitch
