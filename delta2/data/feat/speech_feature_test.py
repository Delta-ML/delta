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
''' speech feature entrypoint unittest'''
import os
from pathlib import Path

import librosa
import numpy as np
import delta.compat as tf
from absl import logging

from delta.data.feat import speech_ops
from delta.data.feat import speech_feature
from delta import PACKAGE_ROOT_DIR


#pylint: disable=too-many-instance-attributes
class SpeechFeatureTest(tf.test.TestCase):
  ''' speech feat entrypoint unittest'''

  def setUp(self):
    super().setUp()
    self.winlen = 0.025
    self.winstep = 0.010
    self.feature_size = 40
    self.sr = 8000  #pylint: disable=invalid-name
    self.nfft = 512
    self.feat_type = 'logfbank'

    package_root = Path(PACKAGE_ROOT_DIR)
    self.wavfile = str(
      package_root.joinpath('data/feat/python_speech_features/english.wav'))
    self.featfile = str(
      package_root.joinpath('data/feat/python_speech_features/english.npy'))

  def tearDown(self):
    ''' tear down '''
    if os.path.exists(self.featfile):
      os.unlink(self.featfile)

  def test_load_wav(self):
    ''' test load wav '''
    sample_rate, audio = speech_feature.load_wav(self.wavfile, sr=self.sr)
    audio_true, sample_rate_true = librosa.load(self.wavfile, sr=self.sr)

    self.assertEqual(sample_rate, sample_rate_true)
    self.assertEqual(sample_rate, self.sr)
    self.assertAllClose(audio, audio_true)

  def test_tf_fbank(self):
    ''' test tensorflow fbank feature interface '''
    speech_feature.extract_feature((self.wavfile),
                                   winlen=self.winlen,
                                   winstep=self.winstep,
                                   sr=self.sr,
                                   feature_size=self.feature_size,
                                   feature_name='fbank')

    feat = np.load(self.featfile)
    logging.info(f"feat : {feat}")
    self.assertEqual(feat.shape, (425, 40, 1))

    with self.cached_session(use_gpu=False, force_gpu=False):
      feat = speech_ops.delta_delta(feat, 2)
      self.assertEqual(feat.eval().shape, (425, 40, 3))

  def test_tf_spec(self):
    ''' test tensorflow spec feature interface '''
    speech_feature.extract_feature((self.wavfile),
                                   winlen=self.winlen,
                                   winstep=self.winstep,
                                   sr=self.sr,
                                   feature_size=self.feature_size,
                                   feature_name='spec')
    feat = np.load(self.featfile)
    self.assertEqual(feat.shape, (425, 129, 1))

    with self.cached_session(use_gpu=False, force_gpu=False):
      feat = speech_ops.delta_delta(feat, 2)
      self.assertEqual(feat.eval().shape, (425, 129, 3))

  def test_tf_delta_detla(self):
    ''' test tensorflow delta delta '''
    speech_feature.extract_feature((self.wavfile),
                                   winlen=self.winlen,
                                   winstep=self.winstep,
                                   sr=self.sr,
                                   feature_size=self.feature_size,
                                   feature_name='fbank')

    feat = np.load(self.featfile)
    self.assertEqual(feat.shape, (425, 40, 1))
    feat = speech_feature.add_delta_delta(feat, 40, order=2)
    self.assertEqual(feat.shape, (425, 40, 3))

  def test_py_extract_feat(self):
    ''' test python fbank with delta-delta interface '''
    speech_feature.extract_feat((self.wavfile),
                                winlen=self.winlen,
                                winstep=self.winstep,
                                sr=self.sr,
                                feature_size=self.feature_size,
                                nfft=self.nfft,
                                feat_type=self.feat_type)
    feat = np.load(self.featfile)
    self.assertEqual(feat.shape, (426, 40))

    feat = speech_feature.delta_delta(feat, sr=self.sr)
    self.assertEqual(feat.shape, (426, 40, 3))

  def test_py_feat_interface(self):
    ''' test python feat interface '''
    sr_out, samples = speech_feature.load_wav(self.wavfile, sr=self.sr)
    self.assertEqual(sr_out, self.sr)

    pspec = speech_feature.powspec_feat(
        samples,
        sr=self.sr,
        nfft=self.nfft,
        winlen=self.winlen,
        winstep=self.winstep)
    self.assertEqual(pspec.shape, (426, 257))

    fbank = speech_feature.fbank_feat(
        pspec, feature_size=self.feature_size, sr=self.sr, nfft=self.nfft)
    self.assertEqual(fbank.shape, (426, 40))

    fbank = speech_feature.delta_delta(fbank, sr=self.sr)
    self.assertEqual(fbank.shape, (426, 40, 3))

    feat_ = speech_feature.extract_fbank(
        samples,
        sr=self.sr,
        winlen=self.winlen,
        winstep=self.winstep,
        feature_size=self.feature_size)
    self.assertEqual(feat_.shape, (426, 40, 3))

    samples = np.pad(samples, [0, 100], mode='constant')
    pspec = speech_feature.powspec_feat(
        samples,
        sr=self.sr,
        nfft=self.nfft,
        winlen=self.winlen,
        winstep=self.winstep)
    self.assertEqual(pspec.shape, (427, 257))
    fbank = speech_feature.fbank_feat(
        pspec, feature_size=self.feature_size, sr=self.sr, nfft=self.nfft)
    fbank = speech_feature.delta_delta(fbank, sr=self.sr)
    self.assertEqual(fbank.shape, (427, 40, 3))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
