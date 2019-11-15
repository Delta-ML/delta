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
"""Solver for NLU joint model in raw tensorflow."""

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation, no-name-in-module

from absl import logging
import delta.compat as tf
from tensorflow_addons.text import crf_decode

from delta.utils.register import registers
from delta.utils.solver.raw_solver import RawSolver


@registers.solver.register
class RawNLUJointSolver(RawSolver):
  """Solver for NLU joint model."""

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """

    transitions = model.transitions
    intent_logits, slots_logits = model.logits
    input_intent_y, input_slots_y = model.input_y

    intent_score = tf.nn.softmax(intent_logits, name="intent_score")
    intent_preds = tf.argmax(intent_logits, axis=-1, name="intent_preds")
    y_intent_ground_truth = tf.argmax(
        input_intent_y, axis=-1, name="y_intent_ground_truth")

    slots_preds, slots_score = crf_decode(slots_logits, transitions,
                                          model.input_x_len)

    slots_preds = tf.identity(slots_preds, name="slots_preds")
    slots_score = tf.identity(slots_score, name="slots_score")
    y_slots_ground_truth = tf.identity(
        input_slots_y, name="y_slots_ground_truth")

    model.preds = intent_preds, slots_preds
    model.score = intent_score, slots_score
    model.y_ground_truth = y_intent_ground_truth, y_slots_ground_truth
    logging.info("Model built.")

  def build_export_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model for export.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    transitions = model.transitions
    intent_logits, slots_logits = model.logits

    intent_score = tf.nn.softmax(intent_logits, name="intent_score")
    intent_preds = tf.argmax(intent_logits, axis=-1, name="intent_preds")

    slots_preds, slots_score = crf_decode(slots_logits, transitions,
                                          model.input_x_len)

    slots_preds = tf.identity(slots_preds, name="slots_preds")
    slots_score = tf.identity(slots_score, name="slots_score")

    model.preds = intent_preds, slots_preds
    model.score = intent_score, slots_score
    model.output_dict = {
        "slots_score": slots_score,
        "slots_preds": slots_preds,
        "intent_score": intent_score,
        "intent_preds": intent_preds
    }
    logging.info("Model built.")
