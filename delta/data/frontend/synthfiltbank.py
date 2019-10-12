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


class Synthfiltbank(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains three optional parameters:window_length(float, default=0.030),
          frame_length(float, default=0.010), sample_rate(float, default=16000.0).
    :return:An object of class HParams, which is a set of hyperparameters as name-value pairs.
    """
    window_length = 0.030
    frame_length = 0.010
    sample_rate = 16000.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, power_spectrum, phase_spectrum, sample_rate=None):
    """
    Implement frequency domain to time domain conversion.
    :param power_spectrum: a float tensor of size (num_frames, num_frequencies).
    :param phase_spectrum: a float tensor of size (num_frames, num_frequencies).
    :param sample_rate: a scalar tensor.
    :return: audio data
    """

    p = self.config
    with tf.name_scope('synthfiltbank'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=float)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=float))
      with tf.control_dependencies([assert_op]):

        audio_data = py_x_ops.synthfiltbank(
            power_spectrum,
            phase_spectrum,
            sample_rate,
            window_length=p.window_length,
            frame_length=p.frame_length)

        return audio_data
