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

# pylint: disable=missing-docstring

import os
from pathlib import Path
from absl import logging
import numpy as np
import delta.compat as tf

# delta
from delta import utils
from delta.data.task.text_cls_task import TextClsTask
from delta.utils.register import import_all_modules_for_register
from delta import PACKAGE_ROOT_DIR


class TextClsTaskTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    import_all_modules_for_register()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_cls_data/text_cls/v1/config/han-cls.yml')

  def tearDown(self):
    ''' tear down '''

  def test_english(self):
    config = utils.load_config(self.config_file)
    class_num = config["data"]["task"]["classes"]["num_classes"]
    task_config = config["data"]["task"]
    task_config["language"] = "english"
    task_config["split_by_space"] = True
    task_config["clean_english"] = True
    data_config = config["data"]
    data_config["train"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/train.english.txt"
    ]
    data_config["eval"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/eval.english.txt"
    ]
    data_config["infer"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/test.english.txt"
    ]
    task_config[
        "text_vocab"] = "egs/mock_text_cls_data/text_cls/v1/data/text_vocab.english.txt"
    task_config["need_shuffle"] = False
    config["model"]["split_token"] = ""
    task_config["preparer"]["reuse"] = False

    task = TextClsTask(config, utils.TRAIN)

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
      logging.debug(res[1][0][:5])
      self.assertAllEqual(res[0][0][:5], [3, 4, 5, 0, 0])
      self.assertEqual(np.shape(res[1]), (32, class_num))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      res = sess.run(input_x, feed_dict={input_sentence: ["All is well."]})
      logging.debug(res[0][:5])
      self.assertAllEqual(res[0][:5], [3, 4, 5, 0, 0])

  # # comment it for no dense data now
  # def test_english_dense(self):
  #   config = utils.load_config(self.config_file)
  #   max_len = config["model"]["net"]["structure"]["max_len"]
  #   class_num = config["data"]["task"]["classes"]["num_classes"]
  #   data_config = config["data"]
  #   task_config = data_config["task"]
  #   task_config["language"] = "chinese"
  #   task_config["split_by_space"] = True
  #   task_config["use_dense"] = True
  #   task_config["dense_input_dim"] = 31
  #   data_config["train"][
  #       "dense_npy"] = "./delta/config/data/text_cls/english/dense_data/ds_train_scale.npy"
  #   data_config["eval"][
  #       "dense_npy"] = "./delta/config/data/text_cls/english/dense_data/ds_eval_scale.npy"
  #   data_config["infer"][
  #       "dense_npy"] = "./delta/config/data/text_cls/english/dense_data/ds_test_scale.npy"
  #
  #   task = TextClsTask(config, utils.TRAIN)
  #
  #   # test offline data
  #   # task.do_pre_process()
  #   data = task.dataset()
  #   self.assertTrue("input_x_dict" in data and
  #                   "input_x" in data["input_x_dict"])
  #   self.assertTrue("input_x_dict" in data and
  #                   "input_dense" in data["input_x_dict"])
  #   self.assertTrue("input_y_dict" in data and
  #                   "input_y" in data["input_y_dict"])
  #   with self.cached_session(use_gpu=False, force_gpu=False) as sess:
  #     sess.run(data["iterator"].initializer, feed_dict=data["init_feed_dict"])
  #     res = sess.run([
  #         data["input_x_dict"]["input_x"], data["input_x_dict"]["input_dense"],
  #         data["input_y_dict"]["input_y"]
  #     ])
  #     logging.debug(res[0][0])
  #     logging.debug(res[1][0])
  #     logging.debug(res[2][0])
  #     self.assertEqual(np.shape(res[0]), (32, max_len))
  #     self.assertEqual(np.shape(res[1]), (32, task_config["dense_input_dim"]))
  #     self.assertEqual(np.shape(res[2]), (32, class_num))
  #
  #   # test online data
  #   export_inputs = task.export_inputs()
  #   self.assertTrue("export_inputs" in export_inputs and
  #                   "input_sentence" in export_inputs["export_inputs"])
  #   input_sentence = export_inputs["export_inputs"]["input_sentence"]
  #   input_x = export_inputs["model_inputs"]["input_x"]
  #   with self.cached_session(use_gpu=False, force_gpu=False) as sess:
  #     res = sess.run(input_x, feed_dict={input_sentence: ["All is well."]})
  #     logging.debug(res[0])
  #     self.assertEqual(np.shape(res[0]), (max_len,))

  def test_chinese_split_by_space(self):
    config = utils.load_config(self.config_file)
    class_num = config["data"]["task"]["classes"]["num_classes"]
    data_config = config["data"]
    task_config = data_config["task"]
    task_config["language"] = "chinese"
    task_config["split_by_space"] = True
    task_config["use_word"] = False
    data_config = config["data"]
    data_config["train"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/train.split_by_space.txt"
    ]
    data_config["eval"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/eval.split_by_space.txt"
    ]
    data_config["infer"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/test.split_by_space.txt"
    ]
    task_config[
        "text_vocab"] = "egs/mock_text_cls_data/text_cls/v1/data/text_vocab.split_by_space.txt"
    task_config["need_shuffle"] = False
    config["model"]["split_token"] = ""
    task_config["preparer"]["reuse"] = False

    task = TextClsTask(config, utils.TRAIN)

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
      logging.debug(res[0][0])
      logging.debug(res[1][0])
      self.assertAllEqual(res[0][0][:5], [2, 3, 0, 0, 0])
      self.assertEqual(np.shape(res[1]), (32, class_num))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      res = sess.run(input_x, feed_dict={input_sentence: ["都 挺好"]})
      logging.debug(res[0][:5])
      logging.debug(np.shape(res[0]))
      self.assertAllEqual(res[0][:5], [2, 3, 0, 0, 0])

  def test_chinese_word(self):
    config = utils.load_config(self.config_file)
    class_num = config["data"]["task"]["classes"]["num_classes"]
    data_config = config["data"]
    task_config = data_config["task"]
    task_config["language"] = "chinese"
    task_config["split_by_space"] = False
    task_config["use_word"] = True
    data_config = config["data"]
    data_config["train"]["paths"] = \
      ["egs/mock_text_cls_data/text_cls/v1/data/train.chinese_word.txt"]
    data_config["eval"]["paths"] = \
      ["egs/mock_text_cls_data/text_cls/v1/data/eval.chinese_word.txt"]
    data_config["infer"]["paths"] = \
      ["egs/mock_text_cls_data/text_cls/v1/data/test.chinese_word.txt"]
    task_config[
        "text_vocab"] = "egs/mock_text_cls_data/text_cls/v1/data/text_vocab.chinese_word.txt"
    task_config["need_shuffle"] = False
    config["model"]["split_token"] = ""
    task_config["preparer"]["reuse"] = False

    task = TextClsTask(config, utils.TRAIN)

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
      logging.debug(res[0][0])
      logging.debug(res[1][0])
      self.assertAllEqual(res[0][0][:5], [2, 0, 0, 0, 0])
      self.assertEqual(np.shape(res[1]), (32, class_num))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]
    shape_op = tf.shape(input_x)

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      res, shape_res = sess.run([input_x, shape_op], feed_dict={input_sentence: ["我很愤怒"]})
      logging.debug(res[0])
      logging.debug(np.shape(res[0]))
      logging.debug(f"shape: {shape_res}")
      self.assertAllEqual(shape_res, [1, 1024])
      self.assertAllEqual(res[0][:5], [4, 5, 0, 0, 0])

  def test_chinese_char(self):
    config = utils.load_config(self.config_file)
    max_len = config["model"]["net"]["structure"]["max_len"]
    class_num = config["data"]["task"]["classes"]["num_classes"]
    data_config = config["data"]
    task_config = data_config["task"]
    task_config["language"] = "chinese"
    task_config["split_by_space"] = False
    task_config["use_word"] = False
    data_config = config["data"]
    data_config["train"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/train.split_by_char.txt"
    ]
    data_config["eval"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/eval.split_by_char.txt"
    ]
    data_config["infer"]["paths"] = [
        "egs/mock_text_cls_data/text_cls/v1/data/test.split_by_char.txt"
    ]
    task_config[
        "text_vocab"] = "egs/mock_text_cls_data/text_cls/v1/data/text_vocab.split_by_char.txt"
    task_config["need_shuffle"] = False
    config["model"]["split_token"] = ""
    task_config["preparer"]["reuse"] = False

    task = TextClsTask(config, utils.TRAIN)

    # test offline data
    data = task.dataset()
    self.assertTrue("input_x_dict" in data and
                    "input_x" in data["input_x_dict"])
    self.assertTrue("input_y_dict" in data and
                    "input_y" in data["input_y_dict"])
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run([
          data["input_x_dict"]["input_x"], data["input_y_dict"]["input_y"],
          data["input_x_len"]
      ])
      logging.debug(res[0][0])
      logging.debug(res[1][0])
      self.assertAllEqual(res[0][0][:5], [2, 3, 4, 0, 0])
      self.assertEqual(np.shape(res[0]), (32, max_len))
      self.assertEqual(np.shape(res[1]), (32, class_num))
      self.assertEqual(np.shape(res[2]), (32,))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      res = sess.run(input_x, feed_dict={input_sentence: ["都挺好"]})
      logging.debug(res[0][:5])
      logging.debug(np.shape(res[0]))
      self.assertEqual(np.shape(res[0]), (max_len,))
      self.assertAllEqual(res[0][:5], [2, 3, 4, 0, 0])

  def test_chinese_with_split_token(self):
    config = utils.load_config(self.config_file)
    max_len = config["model"]["net"]["structure"]["max_len"]
    class_num = config["data"]["task"]["classes"]["num_classes"]
    data_config = config["data"]
    task_config = data_config["task"]
    task_config["language"] = "chinese"
    task_config["split_by_space"] = False
    task_config["use_word"] = True
    data_config = config["data"]
    data_config["train"]["paths"] = \
      ["egs/mock_text_cls_data/text_cls/v1/data/train.split_by_line_mark.txt"]
    data_config["eval"]["paths"] = \
      ["egs/mock_text_cls_data/text_cls/v1/data/eval.split_by_line_mark.txt"]
    data_config["infer"]["paths"] = \
      ["egs/mock_text_cls_data/text_cls/v1/data/test.split_by_line_mark.txt"]
    task_config[
        "text_vocab"] = "egs/mock_text_cls_data/text_cls/v1/data/text_vocab.split_by_line_mark.txt"
    task_config["need_shuffle"] = False
    task_config["preparer"]["reuse"] = False

    task = TextClsTask(config, utils.TRAIN)

    # test offline data
    data = task.dataset()
    self.assertTrue("input_x_dict" in data and
                    "input_x" in data["input_x_dict"])
    self.assertTrue("input_y_dict" in data and
                    "input_y" in data["input_y_dict"])
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      sess.run(data["iterator"].initializer)
      res = sess.run([
          data["input_x_dict"]["input_x"], data["input_y_dict"]["input_y"],
          data["input_x_len"]
      ])
      logging.debug(res[0][0][:10])
      logging.debug(res[1][0])
      self.assertAllEqual(
          res[0][0][:10],
          [2, 0, 0, 0, 6, 2, 0, 0, 8, 0])  #[2,3,0,0,6,2,0,0,8,0]
      self.assertEqual(np.shape(res[0]), (32, max_len))
      self.assertEqual(np.shape(res[1]), (32, class_num))
      self.assertEqual(np.shape(res[2]), (32,))

    # test online data
    export_inputs = task.export_inputs()
    self.assertTrue("export_inputs" in export_inputs and
                    "input_sentence" in export_inputs["export_inputs"])
    input_sentence = export_inputs["export_inputs"]["input_sentence"]
    input_x = export_inputs["model_inputs"]["input_x"]

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      res = sess.run(input_x, feed_dict={input_sentence: ["我很愤怒。|超级生气！"]})
      logging.debug(res[0][:10])
      logging.debug(np.shape(res[0]))
      self.assertEqual(np.shape(res[0]), (max_len,))
      self.assertAllEqual(res[0][:10], [4, 5, 0, 0, 6, 9, 10, 0, 0, 0])


if __name__ == "__main__":
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
