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

import os
import delta.compat as tf
from absl import logging
from pathlib import Path

from delta import utils
from transform.tf_wrapper.ops import py_x_ops
from delta.utils.register import registers
from utils.replace_custom_op_attr_pbtxt import edit_pb_txt


class EditPbtxtTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    main_root = os.environ['MAIN_ROOT']
    self.main_root = Path(main_root)
    config_file = self.main_root.joinpath(
        'delta/config/han-cls-keras/han-cls.yml')
    self.config = utils.load_config(config_file)
    solver_name = self.config['solver']['name']
    self.solver = registers.solver[solver_name](self.config)

  def test_export_model(self):

    export_path_base = self.config["solver"]["service"]["model_path"]
    model_version = self.config["solver"]["service"]["model_version"]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(model_version))
    export_path = os.path.abspath(export_path)

    if not os.path.exists(export_path):
      self.solver.export_model()

    old_paths = [
        "tools/cppjieba/dict/jieba.dict.utf8",
        "tools/cppjieba/dict/hmm_model.utf8", "tools/cppjieba/dict/idf.utf8",
        "tools/cppjieba/dict/stop_words.utf8",
        "tools/cppjieba/dict/user.dict.utf8"
    ]
    old_args = [str(self.main_root.joinpath(p)) for p in old_paths]

    edit_pb_txt(old_args, export_path.decode("utf-8"))

    graph = tf.Graph()
    with self.session(graph) as sess:
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 export_path)

      input_sentence_tensor = graph.get_operation_by_name(
          "input_sentence").outputs[0]

      score_tensor = graph.get_operation_by_name("score").outputs[0]

      score = sess.run(
          score_tensor, feed_dict={input_sentence_tensor: ["你好呀北京"]})
      logging.info("score: {}".format(score))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
