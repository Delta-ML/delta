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
"""The model tests OP of Add_noise_rir_end_to_end """

import os
from pathlib import Path
import delta.compat as tf
from delta.data.frontend.add_noise_end_to_end import AddNoiseEndToEnd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from core.ops import PACKAGE_OPS_DIR


def change_file_path(scp_path, filetype, newfilePath):
  with open(scp_path + filetype, 'r') as f:
    s = f.readlines()
  f.close()
  with open(scp_path + newfilePath, 'w') as f:
    for line in s:
      f.write(scp_path + line)
  f.close()


class AddNoiseEndToEndTest(tf.test.TestCase):
  """
  AddNoiseEndToEnd OP test.
  """

  def test_add_noise_end_to_end(self):

    wav_path = str(Path(PACKAGE_OPS_DIR).joinpath('data/sm1_cln.wav'))

    # reset path of noise && rir
    data_path = str(Path(PACKAGE_OPS_DIR).joinpath('data')) + '/'
    noise_file = data_path + 'noiselist_new.scp'
    change_file_path(data_path, 'noiselist.scp', 'noiselist_new.scp')
    rir_file = data_path + 'rirlist_new.scp'
    change_file_path(data_path, 'rirlist.scp', 'rirlist_new.scp')

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      config = {
          'if_add_noise': True,
          'noise_filelist': noise_file,
          'if_add_rir': True,
          'rir_filelist': rir_file
      }
      noisy_path = wav_path[:-4] + '_noisy.wav'
      add_noise_end_to_end = AddNoiseEndToEnd.params(config).instantiate()
      writewav_op = add_noise_end_to_end(wav_path, noisy_path)
      sess.run(writewav_op)


if __name__ == '__main__':

  tf.test.main()
