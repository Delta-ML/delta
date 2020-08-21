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
"""This model extracts MFCC features per frame."""

import delta.compat as tf
from transform.tf_wrapper.ops import py_x_ops
from delta.utils.hparam import HParams
from transform.tf_wrapper.frontend.base_frontend import BaseFrontend
from transform.tf_wrapper.frontend.fbank import Fbank
from transform.tf_wrapper.frontend.framepow import Framepow
import copy


class Mfcc(BaseFrontend):
  """
  Compute mfcc features of every frame in speech, return a float tensor
  with size (num_channels, num_frames, num_frequencies).
  """

  def __init__(self, config: dict):
    super().__init__(config)
    self.framepow = Framepow(config)
    self.fbank = Fbank(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains fourteen optional parameters.
        --window_length				: Window length in seconds. (float, default = 0.025)
        --frame_length				: Hop length in seconds. (float, default = 0.010)
        --snip_edges				: If True, the last frame (shorter than window_length) will
                              be cutoff. If False, 1 // 2 frame_length data will be padded
                              to data. (bool, default = True)
        ---raw_energy				: If 1, compute frame energy before preemphasis and
                                      windowing. If 2, compute frame energy after
                                      preemphasis and windowing. (int, default = 1)
        --preEph_coeff			    : Coefficient for use in frame-signal preemphasis.
                                      (float, default = 0.97)
        --window_type				: Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria").
                                      (string, default = "povey")
        --remove_dc_offset		    : Subtract mean from waveform on each frame
                                      (bool, default = true)
        --is_fbank					: If true, compute power spetrum without frame energy. If
                                      false, using the frame energy instead of the square of the
                                      constant component of the signal. (bool, default = true)
        --output_type				: If 1, return power spectrum. If 2, return log-power
                                      spectrum. (int, default = 1)
        --upper_frequency_limit		: High cutoff frequency for mel bins (if < 0, offset from
                                      Nyquist) (float, default = 0)
        --lower_frequency_limit		: Low cutoff frequency for mel bins (float, default = 20)
        --filterbank_channel_count	: Number of triangular mel-frequency bins.
                                     (float, default = 23)
        --coefficient_count         : Number of cepstra in MFCC computation.
                                     (int, default = 13)
        --cepstral_lifter           : Constant that controls scaling of MFCCs.
                                     (float, default = 22)
        --use_energy                :Use energy (not C0) in MFCC computation.
                                     (bool, default = True)
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
    cepstral_lifter = 22.0
    coefficient_count = 13
    use_energy = True
    dither = 0.0
    is_log10 = False

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
    hparams.add_hparam('cepstral_lifter', cepstral_lifter)
    hparams.add_hparam('coefficient_count', coefficient_count)
    hparams.add_hparam('use_energy', use_energy)
    hparams.add_hparam('dither', dither)
    hparams.add_hparam('is_log10', is_log10)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
    Caculate mfcc features of audio data.
    :param audio_data: the audio signal from which to compute spectrum.
                       Should be an (1, N) tensor.
    :param sample_rate: the samplerate of the signal we working with.
    :return: A float tensor of size (num_channels, num_frames, num_frequencies)
            containing mfcc features of every frame in speech.
    """
    p = self.config
    with tf.name_scope('mfcc'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))
      with tf.control_dependencies([assert_op]):

        fbank_feats = self.fbank(audio_data, sample_rate)
        sample_rate = tf.cast(sample_rate, dtype=tf.int32)
        shape = tf.shape(fbank_feats)
        nframe = shape[0]
        nfbank = shape[1]
        fbank_feats = tf.reshape(fbank_feats, (1, nframe, nfbank))
        framepow_feats = self.framepow(audio_data, sample_rate)
        mfcc = py_x_ops.mfcc(
            fbank_feats,
            framepow_feats,
            sample_rate,
            use_energy=p.use_energy,
            cepstral_lifter=p.cepstral_lifter,
            coefficient_count=p.coefficient_count)
        return mfcc
