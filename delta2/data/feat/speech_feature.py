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
''' speech feat entrypoint unittest'''
import os

import numpy as np
import delta.compat as tf
from absl import logging

from delta.data.feat import speech_ops
from delta.layers.ops import py_x_ops
from delta.data.feat import python_speech_features as psf

_global_sess = {}


def _get_session(feat_name, graph=None):
  global _global_sess
  sess = None
  if feat_name not in _global_sess:
    assert graph is not None
    sess = tf.Session(graph=graph)
    _global_sess[feat_name] = sess
  else:
    sess = _global_sess[feat_name]
  return sess


def _get_out_tensor_name(tensor_name, output_index):
  return tensor_name + ":" + str(output_index)


def _freq_feat_graph(feat_name, **kwargs):
  winlen = kwargs.get('winlen')
  winstep = kwargs.get('winstep')
  feature_size = kwargs.get('feature_size')
  sr = kwargs.get('sr')  #pylint: disable=invalid-name
  nfft = kwargs.get('nfft')
  del nfft

  assert feat_name in ('fbank', 'spec')

  params = speech_ops.speech_params(
      sr=sr,
      bins=feature_size,
      add_delta_deltas=False,
      audio_frame_length=winlen,
      audio_frame_step=winstep)

  graph = None
  if feat_name == 'fbank':
    # get session
    if feat_name not in _global_sess:
      graph = tf.Graph()
      #pylint: disable=not-context-manager
      with graph.as_default():
        # fbank
        filepath = tf.placeholder(dtype=tf.string, shape=[], name='wavpath')
        waveforms, sample_rate = speech_ops.read_wav(filepath, params)
        del sample_rate
        fbank = speech_ops.extract_feature(waveforms, params)
        # shape must be [T, D, C]
        feat = tf.identity(fbank, name=feat_name)
  elif feat_name == 'spec':
    # magnitude spec
    if feat_name not in _global_sess:
      graph = tf.Graph()
      #pylint: disable=not-context-manager
      with graph.as_default():
        filepath = tf.placeholder(dtype=tf.string, shape=[], name='wavpath')
        waveforms, sample_rate = speech_ops.read_wav(filepath, params)

        spec = py_x_ops.spectrum(
            waveforms[:, 0],
            tf.cast(sample_rate, tf.dtypes.float32),
            output_type=1)  #output_type: 1, power spec; 2 log power spec
        spec = tf.sqrt(spec)
        # shape must be [T, D, C]
        spec = tf.expand_dims(spec, -1)
        feat = tf.identity(spec, name=feat_name)
  else:
    raise ValueError(f"Not support freq feat: {feat_name}.")

  return graph, (_get_out_tensor_name('wavpath',
                                      0), _get_out_tensor_name(feat_name, 0))


#pylint: disable=too-many-locals
def extract_feature(*wavefiles, **kwargs):
  ''' tensorflow fbank feat '''
  dry_run = kwargs.get('dry_run')
  feat_name = 'fbank'
  feat_name = kwargs.get('feature_name')
  assert feat_name

  graph, (input_tensor, output_tensor) = _freq_feat_graph(feat_name, **kwargs)
  sess = _get_session(_get_out_tensor_name(feat_name, 0), graph)

  for wavpath in wavefiles:
    savepath = os.path.splitext(wavpath)[0] + '.npy'
    logging.debug('extract_feat: input: {}, output: {}'.format(
        wavpath, savepath))

    feat = sess.run(output_tensor, feed_dict={input_tensor: wavpath})

    # save feat
    if dry_run:
      logging.info('save feat: path {} shape:{} dtype:{}'.format(
          savepath, feat.shape, feat.dtype))
    else:
      np.save(savepath, feat)


def add_delta_delta(feat, feat_size, order=2):
  ''' add delta detla '''
  feat_name = 'delta_delta'
  graph = None
  # get session
  if feat_name not in _global_sess:
    graph = tf.Graph()
    #pylint: disable=not-context-manager
    with graph.as_default():
      fbank = tf.placeholder(
          dtype=tf.float32, shape=[None, feat_size, 1], name='fbank')
      feat_with_delta_delta = speech_ops.delta_delta(fbank, order=order)
      feat_with_delta_delta = tf.identity(feat_with_delta_delta, name=feat_name)

  sess = _get_session(feat_name, graph)
  feat = sess.run(
      _get_out_tensor_name(feat_name, 0), feed_dict={'fbank:0': feat})
  return feat


#pylint: disable=invalid-name
def load_wav(wavpath, sr=8000):
  '''
  audio:
    np.float32, shape [None], sample in [-1, 1], using librosa.load
    np.int16, shape [None], sample in [-32768, 32767], using scipy.io.wavfile
    np.float32, shape[None, audio_channel], sample int [-1, 1], using tf.DecodeWav

  return
    sr: sample rate
    audio: [-1, 1], same to tf.DecodeWav
  '''
  #from scipy.io import wavfile
  #sample_rate, audio = wavfile.read(wavpath)

  #samples, sample_rate = librosa.load(wavpath, sr=sr)

  feat_name = 'load_wav'
  graph = None
  # get session
  if feat_name not in _global_sess:
    graph = tf.Graph()
    with graph.as_default():
      params = speech_ops.speech_params(sr=sr, audio_desired_samples=-1)
      t_wavpath = tf.placeholder(dtype=tf.string, name="wavpath")
      t_audio, t_sample_rate = speech_ops.read_wav(t_wavpath, params)
      t_audio = tf.identity(t_audio, name="audio")
      t_sample_rate = tf.identity(t_sample_rate, name="sample_rate")

  sess = _get_session(feat_name, graph)
  audio, sample_rate = sess.run([
      _get_out_tensor_name('audio', 0),
      _get_out_tensor_name('sample_rate', 0)
  ],
                                feed_dict={"wavpath:0": wavpath})
  audio = audio[:, 0]

  assert sample_rate == sr, 'sampling rate must be {}Hz, get {}Hz'.format(
      sr, sample_rate)
  return sample_rate, audio


#pylint: disable=invalid-name
def extract_fbank(samples, sr=8000, winlen=0.025, winstep=0.01,
                  feature_size=40):
  ''' extract logfbank with delta and delta-delta
  Return:
      ndarray of shape [nfrmae, feature_size * 3]
  '''
  feat = psf.logfbank(
      samples,
      nfilt=feature_size,
      samplerate=sr,
      winlen=winlen,
      winstep=winstep,
      lowfreq=0,
      highfreq=None,
      preemph=0.97)
  delta = psf.delta(feat, N=2)
  _delta_delta = psf.delta(delta, N=2)
  return np.stack([feat, delta, _delta_delta], axis=-1)


#pylint: disable=invalid-name
def delta_delta(fbank, sr=8000, feature_size=40):
  '''
  params:
    fbank: [nframe, nfbank]
  return : [nframe, nfbank, 3]
  '''
  del sr, feature_size
  assert fbank.ndim == 2
  delta = psf.delta(fbank, N=2)
  _delta_delta = psf.delta(delta, N=2)
  return np.stack([fbank, delta, _delta_delta], axis=-1)


#pylint: disable=too-many-arguments,invalid-name
def fbank_feat(powspec,
               sr=8000,
               feature_size=40,
               nfft=512,
               lowfreq=0,
               highfreq=None):
  ''' return : [nframe, nfbank] '''
  feat = psf.logfbank_from_powspec(
      powspec,
      samplerate=sr,
      nfilt=feature_size,
      nfft=nfft,
      lowfreq=lowfreq,
      highfreq=highfreq,
  )
  return feat


#pylint: disable=too-many-arguments,invalid-name
def powspec_feat(samples,
                 sr=8000,
                 nfft=512,
                 winlen=0.025,
                 winstep=0.01,
                 lowfreq=0,
                 highfreq=None,
                 preemph=0.97):
  ''' return : [nframe, nfft / 2 + 1] '''
  feat = psf.powerspec(
      samples,
      nfft=nfft,
      samplerate=sr,
      winlen=winlen,
      winstep=winstep,
      lowfreq=lowfreq,
      highfreq=highfreq,
      preemph=preemph)
  return feat


#pylint: disable=invalid-name
def freq_resolution(sr, nfft):
  '''
  :param: sr, sample rate
  :param: nfft, fft points
  :return: freq resolution of one point
  '''
  return (sr / 2) / (nfft / 2)


def points(freq, resolution):
  '''
  :params: freq, freq in Hz
  :params: resolution, Hz of one point
  :return: number of points equal to `freq`
  '''
  return freq / resolution


#pylint: disable=too-many-locals
def extract_feat(*args, **kwargs):
  ''' pyfeat, extract feat from utt and dump it '''
  logging.debug("extract_feat : {}".format(kwargs))

  winlen = kwargs.get('winlen')
  winstep = kwargs.get('winstep')
  feature_size = kwargs.get('feature_size')
  sr = kwargs.get('sr')  #pylint: disable=invalid-name
  nfft = kwargs.get('nfft')
  lowfreq = kwargs.get('lowfreq')
  highfreq = kwargs.get('highfreq')
  preemph = kwargs.get('preemph')
  save_feat_path = kwargs.get('save_feat_path')
  dry_run = kwargs.get('dry_run')
  feat_type = kwargs.get('feat_type')

  del lowfreq, preemph

  if save_feat_path and not os.path.exists(save_feat_path):
    os.makedirs(save_feat_path)

  for wavpath in args:
    if save_feat_path:
      filename = os.path.splitext(os.path.split(wavpath)[-1])[0] + '.npy'
      savepath = os.path.join(save_feat_path, filename)
    else:
      savepath = os.path.splitext(wavpath)[0] + '.npy'
    logging.debug('input: {}, output: {}'.format(wavpath, savepath))

    sr_out, samples = load_wav(wavpath, sr=sr)
    del sr_out
    feat = powspec_feat(
        samples, sr=sr, nfft=nfft, winlen=winlen, winstep=winstep)
    logging.debug('apply power spectorgram')

    if feat_type == 'spectrogram':
      # shape: [T, F]
      feat = psf.logpowerspec(feat)
      if highfreq:
        resolution = freq_resolution(sr, nfft)
        ps = int(points(highfreq, resolution))  #pylint: disable=invalid-name
        logging.debug("feat slice: {} {}".format(ps, type(ps)))
        feat = feat[:, :ps]
      logging.debug('apply log power spectorgram')
    elif feat_type == 'logfbank':
      feat = fbank_feat(feat, sr=sr, nfft=nfft, feature_size=feature_size)
      logging.debug('apply fbank spectorgram')
    else:
      raise ValueError("not support feat method")

    feat = feat.astype(np.float32)
    if dry_run:
      logging.info('save feat: path {} shape:{} dtype:{}'.format(
          savepath, feat.shape, feat.dtype))
    else:
      np.save(savepath, feat)
