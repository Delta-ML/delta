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
''' common utils unittest'''

import os
from pathlib import Path
import delta.compat as tf
from delta import utils
from delta import PACKAGE_ROOT_DIR
from delta.data.utils.common_utils import get_file_len

# pylint: disable=invalid-name,too-many-locals,missing-docstring


class CommonUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_seq_label_data/seq-label/v1/config/seq-label-mock.yml')
    self.config = utils.load_config(self.config_file)

  def tearDown(self):
    ''' tear down '''

  def test_get_file_name(self):
    paths = self.config["data"]["train"]["paths"]
    self.assertEqual(get_file_len(paths), 300)


if __name__ == '__main__':
  tf.test.main()
