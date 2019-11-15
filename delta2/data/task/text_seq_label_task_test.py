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
''' text sequence labeling task unittest '''

import os
from pathlib import Path
import numpy as np
import delta.compat as tf
from absl import logging
from delta import utils
from delta.data.task.text_seq_label_task import TextSeqLabelTask
from delta.utils.register import import_all_modules_for_register
from delta import PACKAGE_ROOT_DIR


class TextSeqLabelTaskTest(tf.test.TestCase):
  ''' sequence labeling task test'''

  def setUp(self):
    super().setUp()
    import_all_modules_for_register()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_seq_label_data/seq-label/v1/config/seq-label-mock.yml')

  def tearDown(self):
    ''' tear down '''

  def test_english(self):
    """ test seq label task of english data """
    config = utils.load_config(self.config_file)
    max_len = config["model"]["net"]["structure"]["max_len"]
    config["data"]["task"]["language"] = "english"
    task_config = config["data"]["task"]
    task_config[
        "text_vocab"] = "egs/mock_text_seq_label_data/seq-label/v1/data/text_vocab.txt"
    task_config["need_shuffle"] = False

    task = TextSeqLabelTask(config, utils.TRAIN)

    # test offline data
    data = task.dataset()
    self.assertTrue("input_x_dict" in data and
                    "input_x" in data["input_x_dict"])
    self.assertTrue("input_y_dict" in data and
                    "input_y" in data["input_y_dict"])
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run(
          [data["input_x_dict"]["input_x"], data["input_y_dict"]["input_y"]])
      logging.debug(res[0][0][:5])
      logging.debug(res[1][0])
      self.assertAllEqual(res[0][0][:5], [2, 3, 4, 5, 0])
      self.assertEqual(np.shape(res[0]), (10, max_len))
      self.assertEqual(np.shape(res[1]), (10, max_len))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run(input_x, feed_dict={input_sentence: ["I feel good ."]})
      logging.debug(res[0][:5])
      self.assertAllEqual(res[0][:5], [0, 3, 4, 5, 0])
      self.assertEqual(np.shape(res[0]), (max_len,))


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
