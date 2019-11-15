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
"""Solver for text classification model in raw tensorflow."""

import delta.compat as tf

from delta.utils.register import registers
from delta.utils.solver.raw_solver import RawSolver

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation


@registers.solver.register
class RawMatchSolver(RawSolver):
  """Solver for raw tensorflow model."""

  def __init__(self, config):
    super().__init__(config)
    self.tasktype = config['data']['task']['type']

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    if self.tasktype == "Classification":
      model.score = tf.nn.softmax(model.logits, name="score")
      model.preds = tf.argmax(model.logits, axis=-1)
      model.y_ground_truth = tf.argmax(model.input_y, axis=-1)
    else:
      raise ValueError("%s is not a valid task type."
                       "Must be in `Ranking` and `Classification`." %
                       (self.tasktype))

  def build_export_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    if self.tasktype == "Classification":
      model.score = tf.nn.softmax(model.logits, name="score")
      model.preds = tf.argmax(model.logits, axis=-1)
      model.output_dict = {"score": model.score, "preds": model.preds}
    else:
      raise ValueError("%s is not a valid task type."
                       "Must be in `Ranking` and `Classification`." %
                       (self.tasktype))
