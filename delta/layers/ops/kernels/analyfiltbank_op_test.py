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
''' spectrum Op unit-test '''
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib

# from absl import logging


class AfbOpTest(tf.test.TestCase):
  ''' analysis filter bank op unittest'''

  def setUp(self):
    '''set up'''
    self.wavpath = str(
        Path(os.environ['MAIN_ROOT']).joinpath(
            'delta/layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_afb(self):
    ''' test afb op'''
    with self.session():
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      power_spc, phase_spc = py_x_ops.analyfiltbank(input_data, sample_rate)

      power_spc_true = np.array([
          0.000421823002, 0.000014681223, 0.000088715387, 0.000011405386,
          0.000029108920, 0.000016433882, 0.000009128947, 0.000016150383,
          0.000068095047, 0.000016092306, 0.000088840192, 0.000021255839,
          0.000033152886, 0.000005644561, 0.000012678992, 0.000009685464,
          0.000022561202, 0.000004176219, 0.000032476772, 0.000063007421,
          0.000001721088, 0.000003773108, 0.000012991571, 0.000006143227,
          0.000005361593, 0.000019796202, 0.000012828057, 0.000040009807,
          0.000009260243, 0.000060815764, 0.000036184814, 0.000018079394,
          0.000004533325, 0.000008295409, 0.000033129665, 0.000022150667,
          0.000020058087, 0.000000962711, 0.000017114238, 0.000007549510,
          0.000023227087, 0.000037615722, 0.000007189777, 0.000006701076,
          0.000016871410, 0.000018671506, 0.000006927207, 0.000004177695,
          0.000005777914, 0.000002745287
      ])

      phase_spc_true = np.array([
          3.141592741013, 0.017522372305, 2.614648103714, 1.024240016937,
          -0.082203239202, 0.177630946040, -0.947744905949, 1.557014584541,
          -2.254315614700, -0.327101945877, -2.747241020203, -1.865882754326,
          -2.847117424011, -0.581349492073, -3.014511823654, 2.957268953323,
          1.846585988998, -1.926323652267, -2.718185901642, -2.704042911530,
          -0.473446547985, -2.938575029373, 2.915200233459, -1.540565252304,
          -3.052149772644, 2.665060997009, -2.724275827408, -2.989539623260,
          -2.875509977341, -2.549245357513, 2.585565090179, 1.503721714020,
          1.570051312447, 1.980712175369, 2.068141937256, -1.657162785530,
          2.774835824966, -1.669888973236, -2.816159725189, 3.112393617630,
          -0.539753019810, 2.466773271561, 2.961024999619, -1.002810001373,
          2.275165081024, -2.257984638214, -2.611628055573, -2.753412723541,
          -2.071642875671, -2.972373962402
      ])
      self.assertEqual(tf.rank(power_spc).eval(), 1)
      self.assertEqual(tf.rank(phase_spc).eval(), 1)
      #      logging.info('output1: {}'.format(output_1.eval().flatten()[:50]))
      #      logging.info('output2: {}'.format(output_2.eval().flatten()[:50]))
      self.assertAllClose(power_spc.eval().flatten()[:50], power_spc_true)
      self.assertAllClose(phase_spc.eval().flatten()[:50], phase_spc_true)


if __name__ == '__main__':
  #  logging.set_verbosity(logging.INFO)
  tf.test.main()
