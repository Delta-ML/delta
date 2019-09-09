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
''' Fbank '''

import abc
import numpy as np
from absl import logging
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from delta import utils
from delta.layers.ops import py_x_ops
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend

class FBank(BaseFrontend):

  def __init__(self, config:dict):
   super().__init__(config)

  @classmethod
  def params(cls, config=None):
    ''' set params '''

    sr = 16000
    bins = 40
    dither = True
    use_delta_deltas = True
    cmvn = False
    cmvn = ''

    if config is not None:
      taskconf = config['data']['task']
      audioconf = taskconf['audio']
      sr= taskconf['audio']['sr'],
      bins = audioconf['feature_size'],
      use_delta_deltas = audioconf['add_delta_deltas'],
      cmvn = audioconf['cmvn'],
      cmvn_path = audioconf['cmvn_path']

    hparams = HParams(cls=cls)
    hparams.add_hparam("sample_rate", sr)
    hparams.add_hparam("preemphasis", 0.97)
    if dither:
      hparams.add_hparam("dither", 1.0 / np.iinfo(np.int16).max)
    else:
      hparams.add_hparam("dither", 0.0)
    hparams.add_hparam("frame_length", 0.025)
    hparams.add_hparam("frame_step", 0.010)
    hparams.add_hparam("lower_edge_hertz", 20.0)
    hparams.add_hparam("upper_edge_hertz", sr / 2.0)
    hparams.add_hparam("num_mel_bins", bins)
    hparams.add_hparam("num_zeropad_frames", 0)

    return hparams

  def call(self, waveforms):
    '''waveforms: [samples, audio_channels]
       return: features, [nframes, feat_size]
    '''
    p = self.config
    with tf.variable_scope('feature_extractor'):
      mel_fbanks = self.compute_mel_filterbank_features(
          waveforms,
          sample_rate=p.sample_rate,
          preemphasis=p.preemphasis,
          frame_length=p.frame_length,
          frame_step=p.frame_step,
          lower_edge_hertz=p.lower_edge_hertz,
          upper_edge_hertz=p.upper_edge_hertz,
          num_mel_bins=p.num_mel_bins,
          apply_mask=False)

      # shape: [nframes, nbins, nchannels]
      fbank_size = utils.shape_list(mel_fbanks)
      logging.debug("fbank size : {}".format(fbank_size))

      # Later models like to flatten the two spatial dims. Instead, we add a
      # unit spatial dim and flatten the frequencies and channels.
      feats = tf.concat([
          tf.reshape(mel_fbanks, [fbank_size[0], fbank_size[1]]),
          tf.zeros((p.num_zeropad_frames, fbank_size[1]))
      ], 0)
    return feats  # shape [nframes, featue_size]

  def compute_mel_filterbank_features(self,
                                      waveforms,
                                      sample_rate=16000,
                                      preemphasis=0.97,
                                      frame_length=0.025,
                                      frame_step=0.010,
                                      fft_length=None,
                                      lower_edge_hertz=80.0,
                                      upper_edge_hertz=7600.0,
                                      num_mel_bins=80,
                                      log_noise_floor=1e-3,
                                      apply_mask=True):
    """Implement mel-filterbank extraction using tf ops.
    Args:
      waveforms: float32 tensor with shape [max_len, nchannels]
      sample_rate: sampling rate of the waveform
      preemphasis: waveform high-pass filtering constant
      frame_length: frame length in ms
      frame_step: frame_Step in ms
      fft_length: number of fft bins
      lower_edge_hertz: lowest frequency of the filterbank
      upper_edge_hertz: highest frequency of the filterbank
      num_mel_bins: filterbank size
      log_noise_floor: clip small values to prevent numeric overflow in log
      apply_mask: When working on a batch of samples, set padding frames to zero
    Returns:
      filterbanks: a float32 tensor with shape [nchannles, max_len, num_bins]
    """
    del log_noise_floor, apply_mask
    spectrogram = self.powspec_feat(
        waveforms,
        sr=sample_rate,
        nfft=512 if not fft_length else fft_length,
        winlen=frame_length,
        winstep=frame_step,
        lowfreq=lower_edge_hertz,
        highfreq=upper_edge_hertz,
        preemph=preemphasis)

    # [channels, time, feat_dim] [1, T, D]
    fbank = self.fbank_feat(
        spectrogram,
        sr=sample_rate,
        feature_size=num_mel_bins,
        nfft=512 if not fft_length else fft_length,
        lowfreq=lower_edge_hertz,
        highfreq=upper_edge_hertz)

    # [time, feat_dim]
    fbank = tf.cond(
        tf.equal(tf.rank(fbank), 3),
        true_fn=lambda: fbank[0, :, :],
        false_fn=lambda: fbank)
    return fbank

  def powspec_feat(self,
                   samples,
                   sr=8000,
                   nfft=512,
                   winlen=0.025,
                   winstep=0.010,
                   lowfreq=0,
                   highfreq=None,
                   preemph=0.97):
    '''
    params:
      samples: [nsample, channels]
    returns:
      powspec: power spectrogram, shape [channels, nframe, nfft / 2 + 1] '''
    del nfft
    del lowfreq
    del highfreq
    del preemph

    assert samples.shape[1] == 1, 'samples channel must be 1'
    feat = contrib_audio.audio_spectrogram(
        samples,
        window_size=winlen * sr,
        stride=winstep * sr,
        magnitude_squared=True)
    return feat

  def fbank_feat(self,
                 powspec,
                 sr=8000,
                 feature_size=40,
                 nfft=512,
                 lowfreq=0,
                 highfreq=None):
    ''' powspec: [audio_channels, spectrogram_length, spectrogram_feat_dim]
        return : [auido_chnnels, nframe, nfbank]
    '''
    del nfft

    true_fn = lambda: tf.expand_dims(powspec, 0)
    false_fn = lambda: powspec
    powspec = tf.cond(tf.equal(tf.rank(powspec), 2), true_fn, false_fn)

    feat = py_x_ops.fbank(
        powspec,
        sr,
        filterbank_channel_count=feature_size,
        lower_frequency_limit=lowfreq,
        upper_frequency_limit=highfreq,
    )
    return feat
