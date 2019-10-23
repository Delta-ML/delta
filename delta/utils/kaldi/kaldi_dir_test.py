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
''' speaker task unittest'''
import os

from absl import logging
import delta.compat as tf

from delta.utils.kaldi import kaldi_dir
from delta.utils.kaldi.kaldi_dir_utils import gen_dummy_meta


class KaldiDirTest(tf.test.TestCase):
  ''' Kaldi dir meta data IO test'''

  def setUp(self):
    super().setUp()

  def tearDown(self):
    ''' tear down'''

  def test_property(self):
    ''' test custom properties  '''
    utt = kaldi_dir.Utt()
    self.assertIsNone(utt.wavlen)
    utt.wavlen = 1
    self.assertEqual(utt.wavlen, 1)

  def test_gen_dummy_data(self):
    ''' test dump and load data '''
    num_spk = 5
    num_utt_per_spk = 3
    meta = gen_dummy_meta(num_spk, num_utt_per_spk)
    self.assertEqual(len(meta.spks), num_spk)

  def test_dump_and_load(self):
    ''' test dump and load data '''
    temp_dir = self.get_temp_dir()
    num_spk = 5
    num_utt_per_spk = 3
    meta = gen_dummy_meta(num_spk, num_utt_per_spk)
    meta.dump(temp_dir, True)
    with open(os.path.join(temp_dir, 'feats.scp'), 'r') as fp_in:
      logging.info('feats.scp:\n%s' % (fp_in.read()))
    loaded_meta = kaldi_dir.KaldiMetaData()
    loaded_meta.load(temp_dir)
    self.assertEqual(len(meta.utts), len(loaded_meta.utts))
    for utt_key in meta.utts.keys():
      self.assertIn(utt_key, loaded_meta.utts)
    self.assertEqual(len(meta.spks), len(loaded_meta.spks))
    for spk_key in meta.spks.keys():
      self.assertIn(spk_key, loaded_meta.spks)


if __name__ == '__main__':
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
