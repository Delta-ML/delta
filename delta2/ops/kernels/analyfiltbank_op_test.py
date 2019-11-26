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
from absl import logging
import delta.compat as tf

from delta.layers.ops import py_x_ops
from delta.data import feat as feat_lib
from delta import PACKAGE_ROOT_DIR


class AfbOpTest(tf.test.TestCase):
  ''' analysis filter bank op unittest'''

  def setUp(self):
    super().setUp()
    self.wavpath = str(
        Path(PACKAGE_ROOT_DIR).joinpath(
            'layers/ops/data/sm1_cln.wav'))

  def tearDown(self):
    '''tear down'''

  def test_afb(self):
    ''' test afb op'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      sample_rate, input_data = feat_lib.load_wav(self.wavpath, sr=16000)

      power_spc, phase_spc = py_x_ops.analyfiltbank(input_data, sample_rate)

      power_spc_true = np.array(
          [[
              4.2182300e-04, 3.6964193e-04, 3.9906241e-05, 2.8196722e-05,
              3.3976138e-04, 3.7671626e-04, 2.2727624e-04, 7.2495081e-05,
              4.3451786e-05, 3.4654513e-06
          ],
           [
               1.4681223e-05, 2.8831255e-05, 3.5616580e-05, 3.9359711e-05,
               1.2714787e-04, 1.2794189e-04, 3.6509471e-05, 1.7578101e-05,
               5.9672035e-05, 2.9785692e-06
           ],
           [
               8.8715387e-05, 6.0998322e-05, 2.7695101e-05, 1.6866413e-04,
               4.6845453e-05, 3.3532990e-05, 5.7005627e-06, 5.1852752e-05,
               1.8390550e-05, 8.3459439e-05
           ],
           [
               1.1405386e-05, 1.8942148e-06, 1.6338145e-06, 1.8362705e-05,
               8.4106450e-06, 4.4174294e-06, 3.6533682e-05, 5.0541588e-05,
               1.6701326e-06, 1.8736981e-05
           ],
           [
               2.9108920e-05, 1.6862698e-05, 3.3437627e-05, 6.9332527e-05,
               5.0028186e-05, 5.9426224e-05, 2.1895030e-06, 2.3780794e-06,
               4.7786685e-05, 7.3811811e-05
           ],
           [
               1.6433882e-05, 9.5777386e-07, 2.0980822e-06, 4.8990279e-07,
               1.4232077e-05, 1.5986938e-05, 2.9042780e-05, 1.1719906e-05,
               2.4548817e-06, 5.3594176e-06
           ],
           [
               9.1289467e-06, 9.4249899e-06, 7.4781286e-07, 1.8923520e-05,
               6.5740237e-06, 4.3209452e-06, 3.9396346e-06, 1.2287317e-05,
               4.6807354e-06, 5.8512210e-06
           ],
           [
               1.6150383e-05, 2.6649790e-05, 1.8610657e-05, 2.2872716e-06,
               1.4209920e-05, 2.3279742e-06, 6.6038615e-06, 2.6169775e-05,
               2.8335158e-05, 1.7595910e-06
           ],
           [
               6.8095047e-05, 9.1859045e-05, 2.6713702e-05, 3.0580850e-05,
               1.4539381e-05, 4.2510033e-05, 2.2579852e-05, 1.4843822e-05,
               2.0883192e-05, 6.0624756e-05
           ],
           [
               1.6092306e-05, 1.4245335e-05, 2.4250150e-05, 6.0177539e-05,
               6.7926321e-06, 3.4922948e-07, 2.1843030e-06, 8.5554876e-07,
               2.6831965e-06, 2.0012436e-05
           ]])

      phase_spc_true = np.array(
          [[
              3.1415927, 3.1415927, 3.1415927, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              3.1415927
          ],
           [
               0.01752237, 1.6688037, 1.4971976, 1.4470094, 2.0516894,
               -2.3112175, -0.7115377, 2.9614341, -1.2494497, -0.7055688
           ],
           [
               2.614648, 0.63351387, -2.0660093, 1.7626916, -1.1257634,
               3.017448, -2.892095, -1.2209401, 1.7407895, -1.0281658
           ],
           [
               1.02424, -1.8967879, -0.6139833, 2.587602, 3.0070715, 1.5781559,
               -1.899145, -1.1459525, -0.24284656, -0.8106653
           ],
           [
               -0.08220324, 0.5497215, 1.7031444, -2.8960562, -1.3680246,
               0.4349923, 2.0676146, 1.2389332, 2.6312854, -1.7511902
           ],
           [
               0.17763095, 2.7475302, -0.20671827, 1.0719725, -2.388657,
               1.189566, -1.0643665, 2.5955305, -0.69036585, -0.5287417
           ],
           [
               -0.9477449, -2.7059674, 0.53469753, 1.9289348, 0.24833842,
               0.03517391, -1.4778724, -0.16577117, -1.7509687, -0.46875867
           ],
           [
               1.5570146, -2.9596932, -0.7975963, 3.0060582, -1.038453,
               0.14911443, -1.5873562, 0.7229206, 2.679422, -1.1890441
           ],
           [
               -2.2543156, 0.47845784, -2.8412538, -0.5494534, 1.6583048,
               -1.4567885, 1.0724461, -2.70243, -0.2690962, 1.8831034
           ],
           [
               -0.32710192, 0.01503609, 0.29720783, -0.7409194, -2.183623,
               2.3637679, 0.6405145, 1.4975713, 0.18241015, 2.2659144
           ]])
      self.assertEqual(tf.rank(power_spc).eval(), 2)
      self.assertEqual(tf.rank(phase_spc).eval(), 2)
      logging.info('Shape of power_spc: {}'.format(power_spc.shape))
      logging.info('Shape of phase_spc: {}'.format(phase_spc.shape))
      self.assertAllClose(power_spc.eval().transpose()[:10, :10],
                          power_spc_true)
      self.assertAllClose(phase_spc.eval().transpose()[:10, :10],
                          phase_spc_true)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
