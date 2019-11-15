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
"""Test for raw nlu joint solver."""

import os
from pathlib import Path
from absl import logging
import delta.compat as tf

from delta import utils
from delta.utils.solver.raw_nlu_joint_solver import RawNLUJointSolver
from delta.utils.register import import_all_modules_for_register
from delta import PACKAGE_ROOT_DIR

# pylint: disable=missing-docstring


class RawNLUJointSolverTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file = package_root.joinpath(
        '../egs/mock_text_nlu_joint_data/nlu-joint/v1/config/nlu_joint.yml')
    self.config = utils.load_config(self.config_file)
    import_all_modules_for_register()

  def test_all(self):
    # train and eval
    solver = RawNLUJointSolver(self.config)
    solver.train_and_eval()
    model_path = solver.get_generated_model_path()
    self.assertNotEqual(model_path, None)

    # infer
    solver.first_eval = True
    solver.infer()
    intent_res_file = self.config["solver"]["postproc"][0].get("res_file", "")
    slots_res_file = self.config["solver"]["postproc"][1].get("res_file", "")
    self.assertTrue(os.path.exists(intent_res_file))
    self.assertTrue(os.path.exists(slots_res_file))

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

        intent_score_tensor = graph.get_operation_by_name(
            "intent_score").outputs[0]
        slots_score_tensor = graph.get_operation_by_name(
            "slots_score").outputs[0]

        score = sess.run(
            [intent_score_tensor, slots_score_tensor],
            feed_dict={input_sentence_tensor: ["I am happy in the KFC"]})
        logging.info("score: {}".format(score))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
