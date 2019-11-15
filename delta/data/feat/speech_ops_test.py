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
''' speech feat ops unittest'''
import os
from pathlib import Path
import numpy as np
import delta.compat as tf
from absl import logging

from delta.data.feat.speech_feature import load_wav
from delta.data.feat import speech_ops as tffeat
from delta import PACKAGE_ROOT_DIR


class SpeechOpsFeatTest(tf.test.TestCase):
  ''' test speech feat ops'''

  def setUp(self):
    super().setUp()
    self.sr_true = 8000
    #pylint: disable=invalid-name
    self.hp = tffeat.speech_params(
        sr=self.sr_true,
        bins=40,
        cmvn=False,
        audio_desired_samples=1000,
        add_delta_deltas=False)
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'data/feat/python_speech_features/english.wav'))
    _, self.audio_true = load_wav(self.wavpath, sr=self.sr_true)

  def test_read_wav(self):
    ''' test read wav op '''
    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      # read wav
      audio, sample_rate = tffeat.read_wav(wavfile, self.hp)
      self.assertEqual(sample_rate.eval(), self.sr_true)

      self.assertEqual(audio.eval().shape, (1000, 1))
      self.assertAllEqual(audio.eval()[:, 0], self.audio_true[:1000])

  def test_powspec_feat(self):
    ''' test spectrogram op '''
    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      # read wav
      audio, sample_rate = tffeat.read_wav(wavfile, self.hp)
      del sample_rate
      # spectorgram
      spectrogram = tffeat.powspec_feat(
          audio,
          sr=self.sr_true,
          nfft=None,
          winlen=self.hp.audio_frame_length,
          winstep=self.hp.audio_frame_step,
          lowfreq=self.hp.audio_lower_edge_hertz,
          highfreq=self.hp.audio_upper_edge_hertz,
          preemph=self.hp.audio_preemphasis)

      nfft = int(np.log2(self.hp.audio_frame_length * self.sr_true)) + 1
      nfft = 1 << nfft
      self.assertEqual(spectrogram.eval().shape, (1, 11, int(nfft / 2 + 1)))

  def test_extract_logfbank_with_delta(self):
    ''' test logfbank with delta op'''
    #pylint: disable=invalid-name
    hp = tffeat.speech_params(
        sr=self.sr_true,
        bins=40,
        cmvn=False,
        audio_desired_samples=1000,
        add_delta_deltas=False)

    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      # read wav
      audio, sample_rate = tffeat.read_wav(wavfile, hp)
      del sample_rate
      # fbank with delta delta
      fbank = tffeat.extract_logfbank_with_delta(audio, hp)
      self.assertEqual(fbank.eval().shape, (11, 40, 1))

  def test_extract_feature(self):
    ''' test logfbank with delta, and cmvn '''
    #pylint: disable=invalid-name
    hp = tffeat.speech_params(
        sr=self.sr_true,
        bins=40,
        cmvn=False,
        audio_desired_samples=1000,
        add_delta_deltas=True)

    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      # read wav
      audio, sample_rate = tffeat.read_wav(wavfile, hp)
      del sample_rate

      # fbank with delta delta and cmvn
      feature = tffeat.extract_feature(audio, hp)

      self.assertEqual(feature.eval().shape, (11, 40, 3))

  def test_batch_extract_feature(self):
    ''' test batched feature extraction '''
    #pylint: disable=invalid-name
    hp = tffeat.speech_params(
        sr=self.sr_true,
        bins=40,
        cmvn=False,
        audio_desired_samples=1000,
        add_delta_deltas=True)

    batch_size = 2
    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      # read wav
      audio, sample_rate = tffeat.read_wav(wavfile, hp)
      del sample_rate

      audio = tf.stack([audio] * batch_size)

      # fbank with delta delta and cmvn
      feature = tffeat.batch_extract_feature(audio, hp)

      self.assertEqual(feature.eval().shape, (batch_size, 11, 40, 3))

  def test_compute_mel_filterbank_features(self):
    ''' test logfbank ops'''
    #pylint: disable=invalid-name
    p = tffeat.speech_params(
        sr=self.sr_true,
        bins=40,
        cmvn=False,
        audio_desired_samples=1000,
        add_delta_deltas=False)

    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      audio, sample_rate = tffeat.read_wav(wavfile, self.hp)
      del sample_rate

      feature = tffeat.compute_mel_filterbank_features(
          audio,
          sample_rate=p.audio_sample_rate,
          preemphasis=p.audio_preemphasis,
          frame_length=p.audio_frame_length,
          frame_step=p.audio_frame_step,
          lower_edge_hertz=p.audio_lower_edge_hertz,
          upper_edge_hertz=p.audio_upper_edge_hertz,
          num_mel_bins=p.audio_num_mel_bins,
          apply_mask=False)

      self.assertEqual(feature.eval().shape, (11, 40))

  def test_delta_delta(self):
    ''' test add delta detlas '''
    #pylint: disable=invalid-name
    p = tffeat.speech_params(
        sr=self.sr_true,
        bins=40,
        cmvn=False,
        audio_desired_samples=1000,
        add_delta_deltas=False)

    with self.cached_session(use_gpu=False, force_gpu=False):
      wavfile = tf.constant(self.wavpath)
      audio, sample_rate = tffeat.read_wav(wavfile, self.hp)
      del sample_rate

      feature = tffeat.compute_mel_filterbank_features(
          audio,
          sample_rate=p.audio_sample_rate,
          preemphasis=p.audio_preemphasis,
          frame_length=p.audio_frame_length,
          frame_step=p.audio_frame_step,
          lower_edge_hertz=p.audio_lower_edge_hertz,
          upper_edge_hertz=p.audio_upper_edge_hertz,
          num_mel_bins=p.audio_num_mel_bins,
          apply_mask=False)

      feature = tffeat.delta_delta(feature, order=2)
      self.assertEqual(feature.eval().shape, (11, 40, 3))

  def test_splice(self):
    ''' test batch splice frame '''
    with self.cached_session(use_gpu=False, force_gpu=False):
      feat = tf.ones([1, 3, 2], dtype=tf.float32)

      for l_ctx in range(0, 4):
        for r_ctx in range(0, 4):
          ctx = l_ctx + 1 + r_ctx
          out = tffeat.splice(feat, left_context=l_ctx, right_context=r_ctx)
          self.assertTupleEqual(out.eval().shape, (1, 3, 2 * ctx))
          self.assertAllEqual(out, tf.ones([1, 3, 2 * ctx]))

      with self.assertRaises(ValueError):
        out = tffeat.splice(feat, left_context=-2, right_context=-2).eval()

      with self.assertRaises(ValueError):
        out = tffeat.splice(feat, left_context=2, right_context=-2).eval()

      with self.assertRaises(ValueError):
        out = tffeat.splice(feat, left_context=-2, right_context=2).eval()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
