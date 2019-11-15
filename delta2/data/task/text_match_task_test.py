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
""" text match task unit test """

import os
from pathlib import Path
from absl import logging
import numpy as np
import delta.compat as tf

from delta import utils
from delta.data.task.text_match_task import TextMatchTask
from delta.utils.register import import_all_modules_for_register
from delta import PACKAGE_ROOT_DIR


class TextMatchTaskTest(tf.test.TestCase):
  """text match task test"""

  # pylint: disable=invalid-name
  def setUp(self):
    super().setUp()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_match_data/text_match/v1/config/rnn-match-mock.yml')
    import_all_modules_for_register()

  def tearDown(self):
    ''' tear down '''

  # pylint: disable=too-many-locals

  def test_english(self):
    """ test text match task of english data """
    config = utils.load_config(self.config_file)
    max_seq_len = config["data"]["task"]["max_seq_len"]
    class_num = config["data"]["task"]["classes"]["num_classes"]
    batch_size = config["data"]["task"]["batch_size"]
    data_config = config["data"]
    task_config = data_config["task"]
    task_config["language"] = "english"
    task_config["split_by_space"] = False
    task_config["use_word"] = True
    task_config[
        "text_vocab"] = "egs/mock_text_match_data/text_match/v1/data/text_vocab.txt"
    task_config["need_shuffle"] = False

    # generate_mock_files(config)

    task = TextMatchTask(config, utils.TRAIN)

    # test offline data
    data = task.dataset()
    self.assertTrue("input_x_dict" in data and
                    "input_x_left" in data["input_x_dict"] and
                    "input_x_right" in data["input_x_dict"])
    self.assertTrue("input_y_dict" in data and
                    "input_y" in data["input_y_dict"])
    # with self.cached_session(use_gpu=False, force_gpu=False) as sess:
    #  sess.run(data["iterator"].initializer)
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      sess.run([data["iterator"].initializer, data["iterator_len"].initializer])
      res = sess.run([
          data["input_x_dict"]["input_x_left"],
          data["input_x_dict"]["input_x_right"],
          data["input_y_dict"]["input_y"],
          data["input_x_len"]["input_x_left_len"],
          data["input_x_len"]["input_x_right_len"],
      ])
      logging.debug(res[0][0][:10])
      logging.debug(res[1][0])
      logging.debug(res[2][0])
      logging.debug(res[3])
      logging.debug(res[4])

      self.assertAllEqual(res[0][0][:10], [2, 3, 4, 5, 6, 0, 0, 0, 0, 0])
      self.assertEqual(np.shape(res[0]), (batch_size, max_seq_len))
      self.assertEqual(np.shape(res[1]), (batch_size, max_seq_len))
      self.assertEqual(np.shape(res[2]), (batch_size, class_num))
      self.assertEqual(np.shape(res[3]), (batch_size,))
      self.assertEqual(np.shape(res[4]), (batch_size,))
    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sent_left" in export_inputs["export_inputs"] and
                    "input_sent_right" in export_inputs["export_inputs"])

    input_sent_left = export_inputs["export_inputs"]["input_sent_left"]
    input_sent_right = export_inputs["export_inputs"]["input_sent_right"]
    input_x_left = export_inputs["model_inputs"]["input_x_left"]
    input_x_right = export_inputs["model_inputs"]["input_x_right"]
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # sess.run(data["iterator"].initializer)
      sess.run(data["iterator"].initializer)
      res1, res2 = sess.run(
          [input_x_left, input_x_right],
          feed_dict={
              input_sent_left: ["How should I approach forgiveness?"],
              input_sent_right: ["I got chickenpox as a child."]
          })
      logging.debug(res1[0][:10])
      logging.debug(res2[0][:10])
      self.assertAllEqual(res1[0][:10], [2, 3, 4, 5, 6, 0, 0, 0, 0, 0])
      self.assertAllEqual(res2[0][:10], [4, 7, 8, 9, 10, 11, 0, 0, 0, 0])
      self.assertEqual(np.shape(res1[0]), (max_seq_len,))
      self.assertEqual(np.shape(res2[0]), (max_seq_len,))


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
