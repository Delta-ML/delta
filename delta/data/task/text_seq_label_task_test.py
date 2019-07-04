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
import tensorflow as tf
from absl import logging
from delta import utils
from delta.data.task.text_seq_label_task import TextSeqLabelTask


class TextSeqLabelTaskTest(tf.test.TestCase):
  ''' sequence labeling task test'''

  def setUp(self):
    ''' set up'''
    main_root = os.environ['MAIN_ROOT']
    main_root = Path(main_root)
    self.config_file = main_root.joinpath('egs/mock_text_seq_label_data/config/seq-label-mock.yml')

  def test_english(self):
    """ test seq label task of english data """
    config = utils.load_config(self.config_file)
    max_len = config["model"]["net"]["structure"]["max_len"]
    config["data"]["task"]["language"] = "english"
    # generate_mock_files(config)
    task = TextSeqLabelTask(config, utils.TRAIN)

    # test offline data
    data = task.dataset()
    self.assertTrue("input_x_dict" in data and
                    "input_x" in data["input_x_dict"])
    self.assertTrue("input_y_dict" in data and
                    "input_y" in data["input_y_dict"])
    with self.session() as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run([data["input_x_dict"]["input_x"],
                      data["input_y_dict"]["input_y"]])
      logging.debug(res[0][0])
      logging.debug(res[1][0])
      self.assertEqual(np.shape(res[0]), (10, max_len))
      self.assertEqual(np.shape(res[1]), (10, max_len))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]
    with self.session() as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run(input_x, feed_dict={input_sentence: ["All is well."]})
      logging.debug(res[0])
      self.assertEqual(np.shape(res[0]), (max_len,))

  def test_chinese_split_by_space(self):
    """ test seq label task of chiniese data, split sentences by space"""

    config = utils.load_config(self.config_file)
    max_len = config["model"]["net"]["structure"]["max_len"]
    data_config = config["data"]
    task_config = data_config["task"]
    task_config["language"] = "chinese"
    task_config["split_by_space"] = True
    task_config["use_word"] = False

    # generate_mock_files(config)
    task = TextSeqLabelTask(config, utils.TRAIN)

    # test offline data
    data = task.dataset()
    self.assertTrue("input_x_dict" in data and
                    "input_x" in data["input_x_dict"])
    self.assertTrue("input_y_dict" in data and
                    "input_y" in data["input_y_dict"])
    with self.session() as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run([data["input_x_dict"]["input_x"],
                      data["input_y_dict"]["input_y"],
                      data["input_x_len"]])

      logging.debug(res[0][0])
      logging.debug(res[1][0])
      logging.debug(res[2])

      self.assertEqual(np.shape(res[0]), (10, max_len))
      self.assertEqual(np.shape(res[1]), (10, max_len))
      self.assertEqual(np.shape(res[2]), (10, ))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]

    with self.session() as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run(input_x, feed_dict={input_sentence: ["北 京 文 物"]})
      logging.debug(res[0][:5])
      logging.debug(np.shape(res[0]))
      self.assertEqual(np.shape(res[0]), (max_len,))


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
