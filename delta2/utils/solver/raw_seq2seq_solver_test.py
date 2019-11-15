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
"""Test for raw sequence to sequence solver."""

import os
from pathlib import Path
from absl import logging
import delta.compat as tf

from delta import utils
from delta.utils.solver.raw_seq2seq_solver import RawS2SSolver
from delta.utils.register import import_all_modules_for_register
from delta import PACKAGE_ROOT_DIR

# pylint: disable=missing-docstring


class RawS2SSolverTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_seq2seq_data/seq2seq/v1/config/transformer-s2s.yml')
    self.config = utils.load_config(self.config_file)
    import_all_modules_for_register()

  def tearDown(self):
    ''' tear down '''

  def test_all(self):
    # train and eval
    solver = RawS2SSolver(self.config)
    solver.train_and_eval()
    model_path = solver.get_generated_model_path()
    self.assertNotEqual(model_path, None)

    # infer
    solver.first_eval = True
    solver.infer()
    res_file = self.config["solver"]["postproc"].get("res_file", "")
    self.assertTrue(os.path.exists(res_file))

    # export model
    solver.export_model()

    export_path_base = self.config["solver"]["service"]["model_path"]
    model_version = self.config["solver"]["service"]["model_version"]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(model_version))
    export_path = os.path.abspath(export_path)
    logging.info("Load exported model from: {}".format(export_path))

    # load the model and run
    graph = tf.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
      with self.cached_session(use_gpu=False, force_gpu=False) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                   export_path)

        input_sentence_tensor = graph.get_operation_by_name(
            "input_sentence").outputs[0]

        preds_tensor = graph.get_operation_by_name("preds").outputs[0]

        preds = sess.run(
            preds_tensor,
            feed_dict={
                input_sentence_tensor: [
                    " vice president walter "
                    "mondale was released"
                ]
            })
        logging.info("preds: {}".format(preds))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
