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
''' speech feat ops interface '''
import numpy as np
from absl import logging

#pylint: disable=no-name-in-module
import delta.compat as tf
from tensorflow.python.ops import gen_audio_ops as audio_ops

from delta import utils
from delta.utils.hparam import HParams
from delta.layers.ops import py_x_ops


#pylint: disable=invalid-name,too-many-arguments
def speech_params(sr=16000,
                  bins=40,
                  dither=True,
                  add_delta_deltas=False,
                  audio_desired_samples=-1,
                  audio_frame_length=0.025,
                  audio_frame_step=0.010,
                  cmvn=False,
                  cmvn_path=''):
  ''' speech feat params '''
  p = HParams()
  p.add_hparam("audio_sample_rate", sr)
  if dither:
    p.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
  else:
    p.add_hparam("audio_dither", 0.0)

  p.add_hparam("audio_preemphasis", 0.97)
  p.add_hparam("audio_desired_channels", 1)
  p.add_hparam("audio_desired_samples", audio_desired_samples)
  p.add_hparam("audio_frame_length", audio_frame_length)
  p.add_hparam("audio_frame_step", audio_frame_step)
  p.add_hparam("audio_lower_edge_hertz", 20.0)
  p.add_hparam("audio_upper_edge_hertz", sr / 2.0)
  p.add_hparam("audio_num_mel_bins", bins)
  p.add_hparam("audio_add_delta_deltas", add_delta_deltas)
  p.add_hparam("num_zeropad_frames", 0)
  p.add_hparam("audio_global_cmvn", cmvn)
  p.add_hparam("audio_cmvn_path", cmvn_path)
  return p


def read_wav(wavfile, params):
  '''
     params:
       wavfile: file name
     returns:
       audio: samples of shape [nsample, channels]
       sample_rate: sample rate
  '''
  contents = tf.read_file(wavfile)
  #pylint: disable=no-member
  waveforms = tf.audio.decode_wav(
      contents,
      desired_channels=params.audio_desired_channels,
      desired_samples=params.audio_desired_samples,
  )
  #waveforms = tf.contrib.ffmpeg.decode_audio(
  #  contents,
  #  file_format='wav',
  #  samples_per_second = params.audio_sample_rate,
  #  channel_count=params.audio_channels,
  #)
  #return waveforms[:, 0]
  return waveforms.audio, waveforms.sample_rate


#pylint: disable=too-many-arguments
def powspec_feat(samples,
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

  #pylint: disable=no-member
  feat = audio_ops.audio_spectrogram(
      samples,
      window_size=winlen * sr,
      stride=winstep * sr,
      magnitude_squared=True)
  return feat


def fbank_feat(powspec,
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


def delta_delta(feat, order=2):
  '''
  params:
    feat: a tensor of shape [nframe, nfbank] or [nframe, nfbank, 1]
  return: [nframe, nfbank, 3]
  '''
  feat = tf.cond(
      tf.equal(tf.rank(feat), 3),
      true_fn=lambda: feat[:, :, 0],
      false_fn=lambda: feat)

  shape = tf.shape(feat)
  # [nframe nfbank*3]
  nframe = shape[0]
  nfbank = shape[1]
  delta = py_x_ops.delta_delta(feat, order=order)
  feat_with_delta_delta = tf.reshape(delta, (nframe, nfbank, (order + 1)))
  return feat_with_delta_delta


#pylint: disable=too-many-arguments
def compute_mel_filterbank_features(waveforms,
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
  spectrogram = powspec_feat(
      waveforms,
      sr=sample_rate,
      nfft=512 if not fft_length else fft_length,
      winlen=frame_length,
      winstep=frame_step,
      lowfreq=lower_edge_hertz,
      highfreq=upper_edge_hertz,
      preemph=preemphasis)

  # [channels, time, feat_dim]
  fbank = fbank_feat(
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


def extract_logfbank_with_delta(waveforms, params):
  '''
   params:
     waveforms: float32 tensor with shape [max_len]
  '''
  p = params
  mel_fbanks = compute_mel_filterbank_features(
      waveforms,
      sample_rate=p.audio_sample_rate,
      preemphasis=p.audio_preemphasis,
      frame_length=p.audio_frame_length,
      frame_step=p.audio_frame_step,
      lower_edge_hertz=p.audio_lower_edge_hertz,
      upper_edge_hertz=p.audio_upper_edge_hertz,
      num_mel_bins=p.audio_num_mel_bins,
      apply_mask=False)

  if p.audio_add_delta_deltas:
    mel_fbanks = delta_delta(mel_fbanks)
  else:
    mel_fbanks = tf.expand_dims(mel_fbanks, axis=-1)
  # shape: [nframes, nbins, nchannels]
  return mel_fbanks


def extract_feature(waveforms, params):
  '''waveforms: [samples, audio_channels]
     return: features, [nframes, feat_size, channels]
  '''
  p = params
  with tf.variable_scope('feature_extractor'):
    mel_fbanks = extract_logfbank_with_delta(waveforms, params)
    # shape: [nframes, nbins, nchannels]
    fbank_size = utils.shape_list(mel_fbanks)
    #assert fbank_size[0] == 1
    logging.debug("fbank size : {}".format(fbank_size))

    # This replaces CMVN estimation on data
    if not p.audio_global_cmvn:
      mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
      variance = tf.reduce_mean(
          tf.square(mel_fbanks - mean), keepdims=True, axis=1)
    else:
      assert p.audio_cmvn_path
      mean, variance = utils.load_cmvn(p.audio_cmvn_path)

    var_epsilon = 1e-09
    mel_fbanks = utils.apply_cmvn(mel_fbanks, mean, variance, var_epsilon)

    # Later models like to flatten the two spatial dims. Instead, we add a
    # unit spatial dim and flatten the frequencies and channels.
    feats = tf.concat([
        tf.reshape(mel_fbanks, [fbank_size[0], fbank_size[1], fbank_size[2]]),
        tf.zeros((p.num_zeropad_frames, fbank_size[1], fbank_size[2]))
    ], 0)
  return feats  # shape [nframes, featue_size, chnanels]


def _new_tensor_array(name, size, dtype=None):
  ''' create empty TensorArray which can store size elements.'''
  return tf.TensorArray(dtype, size, name=name)


def batch_extract_feature(waveforms, params):
  ''' waveforms: [batch, samples, audio_channels]
  return: features [batch, nframes, feat_size, channles]
  '''

  def _to_tensor_array(name, v, clear_after_read=None):
    ''' create TensorArray from v, of size batch.'''
    ta = tf.TensorArray(
        v.dtype, batch, name=name, clear_after_read=clear_after_read)
    ta = ta.unstack(v)
    return ta

  def _loop_continue(time, inputs, unused_output_tas):
    del unused_output_tas
    batch = tf.shape(inputs)[0]
    return time < batch

  def _loop_body(time, inputs, output_tas):
    feat = extract_feature(inputs[time, ...], params)
    new_output_tas = output_tas.write(time, feat)
    return (time + 1, inputs, new_output_tas)

  batch = tf.shape(waveforms)[0]
  output_tas = _new_tensor_array('batch_feat', batch, dtype=tf.float32)
  time = tf.constant(0, tf.int32)
  loop_vars = (time, waveforms, output_tas)

  parallel_iterations = 10
  shape_invariants = tf.nest.map_structure(
      lambda t: tf.TensorShape(None), loop_vars)

  (time, inputs, output_tas) = tf.while_loop(
      _loop_continue,
      _loop_body,
      loop_vars=loop_vars,
      shape_invariants=shape_invariants,
      parallel_iterations=parallel_iterations,
      swap_memory=False)
  del inputs

  batch_feats = output_tas.stack()
  return batch_feats


def splice(feat, left_context, right_context):
  '''
  splice frame with context
    param: feat, tf.float32, [batch, time, feat]
    return: feat, tf.float32, [batch, time, feat*(left_context + 1 + right_context)]
    reference:
      https://github.com/kaldi-asr/kaldi/src/feat/feature-functions.cc#L205:6
  '''

  def _loop_continue(time, end_time, context, unused_left_context,
                     right_context, unused_output_tas):
    del unused_output_tas
    del unused_left_context
    return time < end_time

  def _loop_body(time, end_time, context, left_context, right_context,
                 output_tas):
    shape = tf.shape(context)
    B, _, D = shape[0], shape[1], shape[2]
    N = (1 + left_context + right_context) * D

    new_feat = context[:, time:time + left_context + 1 + right_context, :]
    new_feat = tf.reshape(new_feat, [B, N])
    new_output_tas = output_tas.write(time, new_feat)
    return (time + 1, end_time, context, left_context, right_context,
            new_output_tas)

  with tf.control_dependencies([
      tf.assert_greater_equal(left_context, 0),
      tf.assert_greater_equal(right_context, 0)
  ]):
    T = tf.shape(feat)[1]
    output_tas = _new_tensor_array('splice_feat_ta', T, dtype=tf.float32)
    time = tf.constant(0, tf.int32)
    first = tf.tile(feat[:, 0:1, :], [1, left_context, 1])
    last = tf.tile(feat[:, -1:, :], [1, right_context, 1])
    context = tf.concat([first, feat], axis=1)
    context = tf.concat([context, last], axis=1)

    loop_vars = (time, T, context, left_context, right_context, output_tas)

    parallel_iterations = 10
    shape_invariants = tf.nest.map_structure(
        lambda t: tf.TensorShape(None), loop_vars)

    (time, end_time, context, left_context, right_context,
     output_tas) = tf.while_loop(
         _loop_continue,
         _loop_body,
         loop_vars=loop_vars,
         shape_invariants=shape_invariants,
         parallel_iterations=parallel_iterations,
         swap_memory=False)
    del context
    del left_context
    del right_context

    batch_spliced_feats = output_tas.stack()
    batch_spliced_feats = tf.transpose(batch_spliced_feats, [1, 0, 2])
  return batch_spliced_feats
