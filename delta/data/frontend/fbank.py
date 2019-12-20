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
"""This model extracts Fbank features per frame."""

import tensorflow as tf
from core.ops import py_x_ops
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend
from delta.data.frontend.spectrum import Spectrum


class Fbank(BaseFrontend):
  """
  Computing filter banks is applying triangular filters on a Mel-scale to the power
   spectrum to extract frequency bands. Return a float tensor with shape
   (num_channels, num_frames, num_frequencies).
  """

  def __init__(self, config: dict):
    super().__init__(config)
    self.spect = Spectrum(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains thirteen optional parameters:
           --window_length				: Window length in seconds. (float, default = 0.025)
           --frame_length				: Hop length in seconds. (float, default = 0.010)
           --snip_edges				: If true, the last frame (shorter than window_length) will be
                                         cutoff. If ,false 1 // 2 frame_length data will be padded
                                         to data. (bool, default = true)
           ---raw_energy				: If 1, compute frame energy before preemphasis and
                                         windowing. If 2,  compute frame energy after
                                         preemphasis and windowing. (int, default = 1)
           --preeph_coeff				: Coefficient for use in frame-signal preemphasis.
                                        (float, default = 0.97)
           --window_type				: Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria").
                                        (string, default = "povey")
           --remove_dc_offset			: Subtract mean from waveform on each frame.
                                         (bool, default = true)
           --is_fbank					: If true, compute power spetrum without frame energy.
                                         If false, using the frame energy instead of the
                                         square of the constant component of the signal.
                                         (bool, default = true)
           --output_type				: If 1, return power spectrum. If 2, return log-power
                                         spectrum. (int, default = 1)
           --upper_frequency_limit		: High cutoff frequency for mel bins (if <= 0, offset
                                        from Nyquist) (float, default = 0)
           --lower_frequency_limit		: Low cutoff frequency for mel bins (float, default = 20)
           --filterbank_channel_count	: Number of triangular mel-frequency bins.
                                        (float, default = 23)
           --dither			    	: Dithering constant (0.0 means no dither).
                                        (float, default = 1) [add robust to training]
    :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
    """

    upper_frequency_limit = 0.0
    lower_frequency_limit = 20.0
    filterbank_channel_count = 23.0
    window_length = 0.025
    frame_length = 0.010
    output_type = 1
    sample_rate = 16000
    snip_edges = True
    raw_energy = 1
    preeph_coeff = 0.97
    window_type = 'povey'
    remove_dc_offset = True
    is_fbank = True
    dither = 0.0

    hparams = HParams(cls=cls)
    hparams.add_hparam('upper_frequency_limit', upper_frequency_limit)
    hparams.add_hparam('lower_frequency_limit', lower_frequency_limit)
    hparams.add_hparam('filterbank_channel_count', filterbank_channel_count)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('output_type', output_type)
    hparams.add_hparam('sample_rate', sample_rate)
    hparams.add_hparam('snip_edges', snip_edges)
    hparams.add_hparam('raw_energy', raw_energy)
    hparams.add_hparam('preeph_coeff', preeph_coeff)
    hparams.add_hparam('window_type', window_type)
    hparams.add_hparam('remove_dc_offset', remove_dc_offset)
    hparams.add_hparam('is_fbank', is_fbank)
    hparams.add_hparam('dither', dither)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
       Caculate fbank features of audio data.
       :param audio_data: the audio signal from which to compute spectrum.
                          Should be an (1, N) tensor.
       :param sample_rate: [option]the samplerate of the signal we working with,
                            default is 16kHz.
       :return: A float tensor of size (num_frames, num_frequencies, num_channels) containing
               fbank features of every frame in speech.
    """
    p = self.config
    with tf.name_scope('fbank'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

      if p.upper_frequency_limit <= 0:
        p.upper_frequency_limit = p.sample_rate / 2.0 + p.upper_frequency_limit
      elif (p.upper_frequency_limit <= p.lower_frequency_limit) or (
          p.upper_frequency_limit > p.sample_rate / 2.0):
        p.upper_frequency_limit = p.sample_rate / 2.0

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))
      with tf.control_dependencies([assert_op]):

        spectrum = self.spect(audio_data, sample_rate)
        spectrum = tf.expand_dims(spectrum, 0)

        fbank = py_x_ops.fbank(
            spectrum,
            sample_rate,
            upper_frequency_limit=p.upper_frequency_limit,
            lower_frequency_limit=p.lower_frequency_limit,
            filterbank_channel_count=p.filterbank_channel_count)

        fbank = tf.squeeze(fbank, axis=0)
        shape = tf.shape(fbank)
        nframe = shape[0]
        nfbank = shape[1]
        fbank = tf.reshape(fbank, (nframe, nfbank, 1))

        return fbank
