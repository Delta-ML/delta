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
"""This model extracts Fbank && Pitch features per frame."""

import delta.compat as tf
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend
from delta.data.frontend.pitch import Pitch
from delta.data.frontend.fbank import Fbank


class FbankPitch(BaseFrontend):
  """
  Compute Fbank && Pitch features respectively，and concate them. Return
  a tensor with shape (num_frames, dim_features).
  """

  def __init__(self, config: dict):
    super().__init__(config)
    self.fbank = Fbank(config)
    self.pitch = Pitch(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains twenty-nine optional parameters:
          --sample_rate         : Samplerate of the signal we working with.
                                  (int, default = 16000)
          --window_length		    : Window length in seconds. (float, default = 0.025)
          --frame_length			  : Hop length in seconds. (float, default = 0.010)
          --snip_edges				  : If true, the last frame (shorter than window_length) will
                                        be cutoff. If false, 1 // 2 frame_length data will be padded
                                         to data. (bool, default = true)
          ---raw_energy				  : If 1, compute frame energy before preemphasis and
                                        windowing. If 2,  compute frame energy after preemphasis
                                         and windowing. (int, default = 1)
          --preEph_coeff			  : Coefficient for use in frame-signal preemphasis.
                                        (float, default = 0.97)
          --window_type				  : Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria").
                                        (string, default = "povey")
          --remove_dc_offset	      : Subtract mean from waveform on each frame.
                                        (bool, default = true)
          --is_fbank				  : If true, compute power spetrum without frame
                                        energy. If false, using the frame energy instead
                                         of the square of the constant component of the
                                         signal. (bool, default = true)
          --output_type				  : If 1, return power spectrum. If 2, return
                                        log-power spectrum. (int, default = 1)
          --upper_frequency_limit	  : High cutoff frequency for mel bins.
                                        (if <= 0, offset from Nyquist) (float, default = 0)
          --lower_frequency_limit	  : Low cutoff frequency for mel bins.
                                        (float, default = 20)
          --filterbank_channel_count  : Number of triangular mel-frequency bins.
                                        (float, default = 23)
          --dither			    	  : Dithering constant (0.0 means no dither).
                                        (float, default = 1)
            [add robust to training]
          --delta-pitch               : Smallest relative change in pitch that our
                                        algorithm measures. (float, default = 0.005)
          --frames-per-chunk          : Only relevant for offline pitch extraction.
                                        (e.g. compute-kaldi-pitch-feats), you can set it to a
                                        small nonzero value, such as 10, for better feature
                                        compatibility with online decoding (affects energy
                                        normalization in the algorithm) (int, default = 0)
          --lowpass-cutoff            : cutoff frequency for LowPass filter (Hz).
                                        (float, default = 1000)
          --lowpass-filter-width      : Integer that determines filter width of lowpass filter,
                                        more gives sharper filter (int, default = 1)
          --max-f0                    : max. F0 to search for (Hz) (float, default = 400)
          --max-frames-latency        : Maximum number of frames of latency that we allow pitch
                                        tracking to introduce into the feature processing
                                        (affects output only if --frames-per-chunk > 0 and
                                        --simulate-first-pass-online=true (int, default = 0)
          --min-f0                    : min. F0 to search for (Hz) (float, default = 50)
          --nccf-ballast              : Increasing this factor reduces NCCF for quiet frames.
                                        (float, default = 7000)
          --nccf-ballast-online       : This is useful mainly for debug; it affects how the
                                        NCCF ballast is computed. (bool, default = false)
          --penalty-factor            : cost factor for FO change. (float, default = 0.1)
          --preemphasis-coefficient   : Coefficient for use in signal preemphasis (deprecated)
                                        (float, default = 0)
          --recompute-frame           : Only relevant for online pitch extraction, or for
                                        compatibility with online pitch extraction.  A
                                        non-critical parameter; the frame at which we recompute
                                        some of the forward pointers, after revising our
                                        estimate of the signal energy. Relevant
                                        if--frames-per-chunk > 0. (int, default = 500)
          --resample-frequency        : Frequency that we down-sample the signal to. Must be
                                        more than twice lowpass-cutoff (float, default = 4000)
          --simulate-first-pass-online : If true, compute-kaldi-pitch-feats will output features
                                         that correspond to what an online decoder would see in
                                         the first pass of decoding-- not the final version of
                                         the features, which is the default.  Relevant if
                                         --frames-per-chunk > 0 (bool, default = false)
          --soft-min-f0               : Minimum f0, applied in soft way, must not exceed
                                        min-f0 (float, default = 10)
          --upsample-filter-width     : Integer that determines filter width when upsampling
                                        NCCF (int, default = 5)
    :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
    """
    hparams = HParams(cls=cls)

    upper_frequency_limit = 0
    lower_frequency_limit = 20.0
    filterbank_channel_count = 23.0
    window_length = 0.025
    frame_length = 0.010
    raw_energy = 1
    preeph_coeff = 0.97
    window_type = 'povey'
    remove_dc_offset = True
    is_fbank = True
    output_type = 1
    dither = 0.0
    sample_rate = 16000
    snip_edges = True
    preemph_coeff = 0.0
    min_f0 = 50.0
    max_f0 = 400.0
    soft_min_f0 = 10.0
    penalty_factor = 0.1
    lowpass_cutoff = 1000.0
    resample_freq = 4000.0
    delta_pitch = 0.005
    nccf_ballast = 7000.0
    lowpass_filter_width = 1
    upsample_filter_width = 5
    max_frames_latency = 0
    frames_per_chunk = 0
    simulate_first_pass_online = False
    recompute_frame = 500
    nccf_ballast_online = False

    hparams.add_hparam('sample_rate', sample_rate)
    hparams.add_hparam('snip_edges', snip_edges)
    hparams.add_hparam('preemph_coeff', preemph_coeff)
    hparams.add_hparam('dither', dither)
    hparams.add_hparam('min_f0', min_f0)
    hparams.add_hparam('max_f0', max_f0)
    hparams.add_hparam('soft_min_f0', soft_min_f0)
    hparams.add_hparam('penalty_factor', penalty_factor)
    hparams.add_hparam('lowpass_cutoff', lowpass_cutoff)
    hparams.add_hparam('resample_freq', resample_freq)
    hparams.add_hparam('delta_pitch', delta_pitch)
    hparams.add_hparam('nccf_ballast', nccf_ballast)
    hparams.add_hparam('lowpass_filter_width', lowpass_filter_width)
    hparams.add_hparam('upsample_filter_width', upsample_filter_width)
    hparams.add_hparam('max_frames_latency', max_frames_latency)
    hparams.add_hparam('frames_per_chunk', frames_per_chunk)
    hparams.add_hparam('simulate_first_pass_online', simulate_first_pass_online)
    hparams.add_hparam('recompute_frame', recompute_frame)
    hparams.add_hparam('nccf_ballast_online', nccf_ballast_online)
    hparams.add_hparam('upper_frequency_limit', upper_frequency_limit)
    hparams.add_hparam('lower_frequency_limit', lower_frequency_limit)
    hparams.add_hparam('filterbank_channel_count', filterbank_channel_count)
    hparams.add_hparam('window_length', window_length)
    hparams.add_hparam('frame_length', frame_length)
    hparams.add_hparam('output_type', output_type)
    hparams.add_hparam('raw_energy', raw_energy)
    hparams.add_hparam('preeph_coeff', preeph_coeff)
    hparams.add_hparam('window_type', window_type)
    hparams.add_hparam('remove_dc_offset', remove_dc_offset)
    hparams.add_hparam('is_fbank', is_fbank)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, audio_data, sample_rate=None):
    """
    Caculate fbank && pitch(concat) features of wav.
    :param audio_data: the audio signal from which to compute spectrum.
                       Should be an (1, N) tensor.
    :param sample_rate: the samplerate of the signal we working with.
    :return: A tensor with shape (num_frames, dim_features), containing
            fbank && pitch feature of every frame in speech.
    """

    p = self.config
    with tf.name_scope('fbank_pitch'):

      if sample_rate == None:
        sample_rate = tf.constant(p.sample_rate, dtype=tf.int32)

      assert_op = tf.assert_equal(
          tf.constant(p.sample_rate), tf.cast(sample_rate, dtype=tf.int32))
      with tf.control_dependencies([assert_op]):

        fbank_feats = tf.squeeze(self.fbank(audio_data, sample_rate))
        pitch_feats = tf.squeeze(self.pitch(audio_data, sample_rate))
        fbank_pitch_feats = tf.concat([fbank_feats, pitch_feats], 1)

        return fbank_pitch_feats
