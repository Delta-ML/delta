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
""""This model extracts framepow features per frame."""

import delta.compat as tf

from core.ops import py_x_ops
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend


class Framepow(BaseFrontend):
  """
  Compute power of every frame in speech. Return a float tensor with
  shape (1 * num_frames).
  """

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains five optional parameters:
        --sample_rate       : Waveform data sample frequency (must match the waveform
                             file, if specified there). (float, default = 16000)
        --window_length		 : Window length in seconds. (float, default = 0.025)
        --frame_length		 : Hop length in seconds. (float, default = 0.010)
        --snip_edges			 : If True, the last frame (shorter than window_length)
                              will be cutoff. If False, 1 // 2 frame_length data will
                              be padded to data. (int, default = True)
        --remove_dc_offset : Subtract mean from waveform on each frame (bool, default = true)
    :return:An object of class HParams, which is a set of hyperparameters as name-value pairs.
    """

    window_length = 0.025
    frame_length = 0.010
    snip_edges = True
    remove_dc_offset = True
    sample_rate = 16000

    hparams = HParams(cls=cls)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('snip_edges', snip_edges)
    hparams.add_hparam('remove_dc_offset', remove_dc_offset)
    hparams.add_hparam('sample_rate', sample_rate)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
    Caculate power of every frame in speech.
    :param audio_data: the audio signal from which to compute spectrum.
                       Should be an (1, N) tensor.
    :param sample_rate: [option]the samplerate of the signal we working with,
                        default is 16kHz.
    :return:A float tensor of size (1 * num_frames) containing power of every
            frame in speech.
    """

    p = self.config
    with tf.name_scope('framepow'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))
      with tf.control_dependencies([assert_op]):

        sample_rate = tf.cast(sample_rate, dtype=float)
        framepow = py_x_ops.frame_pow(
            audio_data,
            sample_rate,
            snip_edges=p.snip_edges,
            remove_dc_offset=p.remove_dc_offset,
            window_length=p.window_length,
            frame_length=p.frame_length)

        return tf.squeeze(framepow)
