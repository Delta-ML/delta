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
"""Solver for sequence labeling model in raw tensorflow."""

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation, no-name-in-module

import delta.compat as tf
from tensorflow_addons.text import crf_decode
from delta.utils.register import registers
from delta.utils.solver.raw_solver import RawSolver


@registers.solver.register
class RawSeqLabelSolver(RawSolver):
  """Solver for raw tensorflow model."""

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    model.preds, score = crf_decode(model.logits, model.transitions,
                                    model.input_x_len)

    model.score = tf.identity(score, name="score")
    model.y_ground_truth = model.input_y

  def build_export_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    model.preds, score = crf_decode(model.logits, model.transitions,
                                    model.input_x_len)

    model.score = tf.identity(score, name="score")
    model.output_dict = {"score": model.score, "preds": model.preds}
