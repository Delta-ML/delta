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
"""Solver for pretrain text classification model in raw tensorflow."""

import re
import logging
import delta.compat as tf
from delta.utils.register import registers
from delta.utils.solver.raw_solver import RawSolver

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation


@registers.solver.register
class RawPretrainClassSolver(RawSolver):
  """Solver for raw tensorflow model."""

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """

    model.score = tf.nn.softmax(model.logits, name="score")
    model.preds = tf.argmax(model.logits, axis=-1)
    if hasattr(model, "input_y"):
      model.y_ground_truth = tf.argmax(model.input_y, axis=-1)
    model.output_dict = {"score": model.score, "preds": model.preds}

    if model.use_pretrained_model:
      self.initialize_pretrained_model_variables(model.pretrained_model_path,
                                                 model.pretrained_model_mode)

  def get_assignment_map_from_checkpoint(self, all_variables, init_checkpoint):
    """
    Get the map of the current variables and init checkpoint variables.
    """
    assignment_map = {}
    name_to_var = {}
    init_set = set()
    for var in all_variables:
      name = var.name
      m = re.match("^(.*):\\d+$", name)
      if m is not None:
        name = m.group(1)
      name_to_var[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    for name, var_shape in init_vars:
      for k, v in name_to_var.items():
        if re.findall(name + '$', k):
          assignment_map[name] = name_to_var[k]
          init_set.add(k)
    return assignment_map, init_set

  def remove_trainable_variables(self, init_set):
    """
    Make the variables of the pretrained model untrainable
    """

    variables_to_untrain = list()
    trainable_collection = tf.get_collection_ref(
        tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in trainable_collection:
      if var.name in init_set:
        variables_to_untrain.append(var)

    for var in variables_to_untrain:
      trainable_collection.remove(var)

  def initialize_pretrained_model_variables(self, pretrained_model_path,
                                            pretrained_model_mode):
    """
    Initialize the variables of the pretrained model
    according to fine-tune of feature mode
    """
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    init_checkpoint = pretrained_model_path
    pretrained_assignment_map, init_set = self.get_assignment_map_from_checkpoint(
        all_variables, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, pretrained_assignment_map)

    if pretrained_model_mode == "feature":
      self.remove_trainable_variables(init_set)
