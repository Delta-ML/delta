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
''' metrics utils unittest '''
import os
from pathlib import Path
import delta.compat as tf
from delta import utils
from delta.utils.postprocess.postprocess_utils import ids_to_sentences
from delta import PACKAGE_ROOT_DIR


class PostprocessUtilsTest(tf.test.TestCase):
  ''' metrics utils unittest'''

  def setUp(self):
    super().setUp()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_seq_label_data/seq-label/v1/config/seq-label-mock.yml')

  def tearDown(self):
    ''' tear down '''

  def test_ids_to_sentences(self):
    ''' test ids_to_sentences function '''
    config = utils.load_config(self.config_file)
    ids = [[2, 3, 1]]
    vocab_file_path = config["data"]["task"]["label_vocab"]
    sents = ids_to_sentences(ids, vocab_file_path)
    self.assertAllEqual(sents, [["I-PER", "B-LOC", "B-PER"]])


if __name__ == "__main__":
  tf.test.main()
