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
''' speech feature in tensorflow'''
import functools

import numpy as np
import delta.compat as tf
import scipy.signal
#pylint: disable=ungrouped-imports,no-name-in-module

from delta import utils
from delta.utils.hparam import HParams


def add_delta_deltas(filterbanks, name=None):
  """Compute time first and second-order derivative channels.
  Args:
    filterbanks: float32 tensor with shape [batch_size, len, num_bins, 1]
    name: scope name
  Returns:
    float32 tensor with shape [batch_size, len, num_bins, 3]
  """
  delta_filter = np.array([2, 1, 0, -1, -2])
  delta_delta_filter = scipy.signal.convolve(delta_filter, delta_filter, "full")
  delta_filter_stack = np.array(
      [[0] * 4 + [1] + [0] * 4, [0] * 2 + list(delta_filter) + [0] * 2,
       list(delta_delta_filter)],
      dtype=np.float32).T[:, None, None, :]

  delta_filter_stack /= np.sqrt(
      np.sum(delta_filter_stack**2, axis=0, keepdims=True))

  filterbanks = tf.nn.conv2d(
      filterbanks,
      delta_filter_stack, [1, 1, 1, 1],
      "SAME",
      data_format="NHWC",
      name=name)
  return filterbanks


#pylint: disable=too-many-arguments,too-many-locals
def compute_mel_filterbank_features(waveforms,
                                    sample_rate=16000,
                                    dither=1.0 / np.iinfo(np.int16).max,
                                    preemphasis=0.97,
                                    frame_length=25,
                                    frame_step=10,
                                    fft_length=None,
                                    window_fn=functools.partial(
                                        tf.signal.hann_window,
                                        periodic=True),
                                    lower_edge_hertz=80.0,
                                    upper_edge_hertz=7600.0,
                                    num_mel_bins=80,
                                    log_noise_floor=1e-3,
                                    apply_mask=True):
  """Implement mel-filterbank extraction using tf ops.
  Args:
    waveforms: float32 tensor with shape [batch_size, max_len]
    sample_rate: sampling rate of the waveform
    dither: stddev of Gaussian noise added to waveform to prevent quantization
      artefacts
    preemphasis: waveform high-pass filtering constant
    frame_length: frame length in ms
    frame_step: frame_Step in ms
    fft_length: number of fft bins
    window_fn: windowing function
    lower_edge_hertz: lowest frequency of the filterbank
    upper_edge_hertz: highest frequency of the filterbank
    num_mel_bins: filterbank size
    log_noise_floor: clip small values to prevent numeric overflow in log
    apply_mask: When working on a batch of samples, set padding frames to zero
  Returns:
    filterbanks: a float32 tensor with shape [batch_size, len, num_bins, 1]
  """
  #  is a complex64 Tensor representing the short-time Fourier
  # Transform of each signal in . Its shape is
  # [batch_size, ?, fft_unique_bins]
  # where fft_unique_bins = fft_length // 2 + 1

  # Find the wave length: the largest index for which the value is !=0
  # note that waveforms samples that are exactly 0.0 are quite common, so
  # simply doing sum(waveforms != 0, axis=-1) will not work correctly.
  wav_lens = tf.reduce_max(
      tf.expand_dims(tf.range(tf.shape(waveforms)[1]), 0) *
      tf.to_int32(tf.not_equal(waveforms, 0.0)),
      axis=-1) + 1
  if dither > 0:
    waveforms += tf.random_normal(tf.shape(waveforms), stddev=dither)
  if preemphasis > 0:
    waveforms = waveforms[:, 1:] - preemphasis * waveforms[:, :-1]
    wav_lens -= 1
  frame_length = int(frame_length * sample_rate / 1e3)
  frame_step = int(frame_step * sample_rate / 1e3)
  if fft_length is None:
    fft_length = int(2**(np.ceil(np.log2(frame_length))))

  stfts = tf.signal.stft(
      waveforms,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length,
      window_fn=window_fn,
      pad_end=True)

  stft_lens = (wav_lens + (frame_step - 1)) // frame_step
  masks = tf.to_float(
      tf.less_equal(
          tf.expand_dims(tf.range(tf.shape(stfts)[1]), 0),
          tf.expand_dims(stft_lens, 1)))

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, 257].
  magnitude_spectrograms = tf.abs(stfts)

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
  linear_to_mel_weight_matrix = (
      tf.signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                    num_spectrogram_bins,
                                                    sample_rate,
                                                    lower_edge_hertz,
                                                    upper_edge_hertz))
  mel_spectrograms = tf.tensordot(magnitude_spectrograms,
                                  linear_to_mel_weight_matrix, 1)
  # Note: Shape inference for tensordot does not currently handle this case.
  mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_sgram = tf.log(tf.maximum(log_noise_floor, mel_spectrograms))

  if apply_mask:
    log_mel_sgram *= tf.expand_dims(tf.to_float(masks), -1)

  return tf.expand_dims(log_mel_sgram, -1, name="mel_sgrams")


def read_wav(wavfile, params):
  ''' samples of shape [nsample] '''
  contents = tf.read_file(wavfile)
  #pylint: disable=no-member
  waveforms = tf.audio.decode_wav(
      contents,
      desired_channels=params.audio_channels,
      #desired_samples=params.audio_sample_rate,
  )
  #waveforms = tf.contrib.ffmpeg.decode_audio(
  #  contents,
  #  file_format='wav',
  #  samples_per_second = params.audio_sample_rate,
  #  channel_count=params.audio_channels,
  #)
  #return waveforms[:, 0]
  return tf.squeeze(waveforms.audio, axis=-1)


#pylint: disable=invalid-name,too-many-arguments
def speech_params(sr=16000,
                  bins=40,
                  dither=True,
                  use_delta_deltas=True,
                  cmvn=False,
                  cmvn_path=''):
  ''' feat params '''
  p = HParams()
  p.add_hparam("audio_sample_rate", sr)
  p.add_hparam("audio_channels", 1)
  p.add_hparam("audio_preemphasis", 0.97)
  if dither:
    p.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
  else:
    p.add_hparam("audio_dither", 0.0)
  p.add_hparam("audio_frame_length", 25.0)
  p.add_hparam("audio_frame_step", 10.0)
  p.add_hparam("audio_lower_edge_hertz", 20.0)
  p.add_hparam("audio_upper_edge_hertz", sr / 2.0)
  p.add_hparam("audio_num_mel_bins", bins)
  p.add_hparam("audio_add_delta_deltas", use_delta_deltas)
  p.add_hparam("num_zeropad_frames", 0)
  p.add_hparam("audio_global_cmvn", cmvn)
  p.add_hparam("audio_cmvn_path", cmvn_path)
  return p


#pylint: disable=invalid-name
def extract_logfbank_with_delta(waveforms, params):
  ''' extract logfbank with delta detla '''
  p = params
  #waveforms = tf.expand_dims(waveforms, 0) # add batch_size dim
  mel_fbanks = compute_mel_filterbank_features(
      waveforms,
      sample_rate=p.audio_sample_rate,
      dither=p.audio_dither,
      preemphasis=p.audio_preemphasis,
      frame_length=p.audio_frame_length,
      frame_step=p.audio_frame_step,
      lower_edge_hertz=p.audio_lower_edge_hertz,
      upper_edge_hertz=p.audio_upper_edge_hertz,
      num_mel_bins=p.audio_num_mel_bins,
      apply_mask=False)
  if p.audio_add_delta_deltas:
    mel_fbanks = add_delta_deltas(mel_fbanks)
  # shape: [batch, nframes, nbins, nchannels]
  return mel_fbanks


#pylint: disable=invalid-name
def extract_feature(waveforms, params):
  '''extract fbank with delta-delta and do cmvn
     waveforms: [batch, samples]
  '''
  p = params
  with tf.variable_scope('feature_extractor'):
    mel_fbanks = extract_logfbank_with_delta(waveforms, params)
    # shape: [1, nframes, nbins, nchannels]
    fbank_size = utils.shape_list(mel_fbanks)
    #assert fbank_size[0] == 1

    # This replaces CMVN estimation on data
    if not p.audio_global_cmvn:
      mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
      variance = tf.reduce_mean(
          tf.square(mel_fbanks - mean), keepdims=True, axis=1)
    else:
      assert p.audio_cmvn_path, p.audio_cmvn_path
      mean, variance = utils.load_cmvn(p.audio_cmvn_path)

    var_epsilon = 1e-09
    mel_fbanks = utils.apply_cmvn(mel_fbanks, mean, variance, var_epsilon)

    # Later models like to flatten the two spatial dims. Instead, we add a
    # unit spatial dim and flatten the frequencies and channels.
    batch_size = fbank_size[0]
    feats = tf.concat([
        tf.reshape(mel_fbanks,
                   [batch_size, fbank_size[1], fbank_size[2], fbank_size[3]]),
        tf.zeros(
            (batch_size, p.num_zeropad_frames, fbank_size[2], fbank_size[3]))
    ], 1)
  return feats  # shape [batch_size, nframes, featue_size, chnanels]
