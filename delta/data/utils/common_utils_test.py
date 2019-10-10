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
import tempfile
import tensorflow as tf
import numpy as np
from delta import utils
from delta.data.utils.common_utils import load_seq_label_raw_data
from delta.data.utils.common_utils import process_one_label_dataset
from delta.data.utils.common_utils import process_multi_label_dataset
from delta.data.utils.common_utils import get_file_len

# pylint: disable=invalid-name,too-many-locals,missing-docstring


class CommonUtilsTest(tf.test.TestCase):

  def setUp(self):
    ''' set up '''
    main_root = os.environ['MAIN_ROOT']
    main_root = Path(main_root)
    self.config_file = main_root.joinpath(
        'egs/mock_text_seq_label_data/seq-label/v1/config/seq-label-mock.yml')
    self.config = utils.load_config(self.config_file)
    vocab_label = [
        "O\t0", "B-PER\t1", "I-PER\t2", "B-LOC\t3", "I-LOC\t4", "B-ORG5\t5",
        "I-ORG\t6", "B-MISC\t7", "I-MISC\t8"
    ]
    vocab_label_filepath = tempfile.mktemp(suffix='label_vocab.txt')
    with open(vocab_label_filepath, mode='w', encoding='utf-8') as fobj:
      for token in vocab_label:
        fobj.write(token)
        fobj.write('\n')
    self.config["data"]["task"]["label_vocab"] = vocab_label_filepath

  def tearDown(self):
    ''' tear down '''

  def test_load_seq_label_raw_data(self):
    mode = utils.TRAIN
    paths = self.config["data"]["train"]["paths"]
    text, label = load_seq_label_raw_data(paths, mode)
    self.assertEqual(text[0], "i feel good .")
    self.assertEqual(label[0], "O O O O")

  def test_process_one_label_dataset(self):
    label = ["O", "O", "O", "I-MISC"]
    label_filepath = tempfile.mktemp(suffix='label_file_for_unitest.txt')
    with open(label_filepath, mode='w', encoding='utf-8') as fobj:
      for token in label:
        fobj.write(token)
        fobj.write('\n')
    label_ds = tf.data.TextLineDataset(label_filepath)
    true_res = [0, 0, 0, 8]
    label_ds = process_one_label_dataset(label_ds, self.config)

    iterator = label_ds.make_initializable_iterator()
    label_res = iterator.get_next()

    with tf.Session() as sess:
      sess.run(iterator.initializer)
      for i in range(len(label)):
        self.assertEqual(np.argmax(sess.run(label_res)), true_res[i])

  def test_process_multi_label_dataset(self):
    label = ["O I-MISC I-MISC", "O B-MISC I-MISC"]
    label_filepath = tempfile.mktemp(suffix='label_file_for_unitest.txt')
    with open(label_filepath, mode='w', encoding='utf-8') as fobj:
      for token in label:
        fobj.write(token)
        fobj.write('\n')
    label_ds = tf.data.TextLineDataset(label_filepath)
    true_res = [[0, 8, 8], [0, 7, 8]]
    label_ds = process_multi_label_dataset(label_ds, self.config)
    iterator = label_ds.make_initializable_iterator()
    label_res = iterator.get_next()

    with tf.Session() as sess:
      sess.run(iterator.initializer)
      for i in range(len(label)):
        self.assertEqual(list(sess.run(label_res)[:3]), true_res[i])

  def test_get_file_name(self):
    paths = self.config["data"]["train"]["paths"]
    self.assertEqual(get_file_len(paths), 300)


if __name__ == '__main__':
  tf.test.main()
