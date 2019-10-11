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
"""Solver for text sequence to sequence model in raw tensorflow."""

import math

import numpy as np
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils import metrics
from delta.utils.register import registers
from delta.utils.solver.raw_solver import RawSolver

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation


@registers.solver.register
class RawS2SSolver(RawSolver):
  """Solver for raw tensorflow model."""

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    if model.mode != utils.INFER:
      model.score = tf.nn.softmax(model.logits, name="score")
      model.preds = tf.argmax(model.logits, axis=-1)
      model.output_dict = {"score": model.score, "preds": model.preds}
    else:
      model.preds = model.logits
      model.output_dict = {"preds": model.preds}
    if hasattr(model, "input_y"):
      model.y_ground_truth = model.input_y

  def build_export_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    model.preds = tf.identity(model.logits, name="preds")
    model.output_dict = {"preds": model.preds}

  def build(self, mode: str):
    """Build the model for training, eval and infer."""
    inputs = self.input_fn(mode)

    self.config["model"]["is_infer"] = mode == utils.INFER

    model = self.model_fn()
    training = mode == utils.TRAIN
    model.logits = model(inputs["input_x_dict"], training=training)
    model.iterator = inputs["iterator"]
    model.input_x_dict = inputs["input_x_dict"]
    model.input_x_len = inputs["input_x_len"]
    model.mode = mode
    loss_fn = self.get_loss_fn()
    if mode != utils.INFER or not self.infer_no_label:
      input_y = inputs["input_y_dict"]["input_y"]
      model.input_y = input_y

    if mode != utils.INFER:
      input_y_len = inputs["input_y_len"]
      model.loss = loss_fn(
          labels=model.input_y,
          logits=model.logits,
          input_length=model.input_x_len,
          label_length=input_y_len,
          name="loss",
      )
      model.loss_op = model.loss
      logging.info("model.loss done")

    # output related
    self.build_output(model)
    return model

  def build_export_model(self):
    """Build the model for export."""
    mode = utils.INFER
    self.config["model"]["is_infer"] = mode == utils.INFER
    export_inputs = self.export_input(mode)

    model = self.model_fn()
    training = mode == utils.TRAIN
    model.logits = model(export_inputs["model_inputs"], training=training)
    model.model_inputs = export_inputs["model_inputs"]
    model.export_inputs = export_inputs["export_inputs"]
    model.input_x_len = export_inputs["model_inputs"]["input_x_len"]
    # output related
    self.build_export_output(model)
    return model

  def eval_or_infer_core(self, model, mode):  # pylint: disable=too-many-locals, too-many-branches
    """The core part of evaluation."""
    model_path = self.get_model_path(mode)
    if model_path is None:
      logging.warning("model_path is None!")
      return

    with model.sess.graph.as_default():
      model.saver.restore(model.sess, save_path=model_path)
      if self.first_eval:
        model.sess.run(tf.tables_initializer())
        self.first_eval = False
      model.sess.run(model.iterator.initializer)

      # Evaluating loop.
      total_loss = 0.0
      data_size = self.config["data"]['{}_data_size'.format(mode)]
      num_batch_every_epoch = int(math.ceil(data_size / self.batch_size))

      y_ground_truth = []
      y_preds = []

      for i in range(num_batch_every_epoch):

        if mode == utils.EVAL:
          loss_val, \
          batch_preds, \
          batch_y_ground_truth = model.sess.run(
              [model.loss, model.preds, model.y_ground_truth])
        elif not self.infer_no_label:
          batch_preds, \
          batch_y_ground_truth = model.sess.run(
            [model.preds, model.y_ground_truth])
        else:
          batch_preds = model.sess.run([model.preds])
          batch_preds = batch_preds[0]

        if mode == utils.EVAL:
          total_loss += loss_val
          y_preds.append([preds for preds in batch_preds])
        else:
          end_id = (i + 1) * self.batch_size

          if data_size < end_id:
            act_end_id = self.batch_size - end_id + data_size
            batch_preds = batch_preds[:act_end_id]
            if not self.infer_no_label:
              batch_y_ground_truth = batch_y_ground_truth[:act_end_id]
          y_preds.extend([preds for preds in batch_preds])

          if not self.infer_no_label:
            y_ground_truth.extend(
                [ground_truth for ground_truth in batch_y_ground_truth])

        if i % 10 == 0 or i == num_batch_every_epoch - 1:
          logging.info("Evaluation rate of "
                       "progress: [ {:.2%} ]".format(
                           i / (num_batch_every_epoch - 1)))

      if mode == utils.EVAL:
        logging.info("Evaluation Average Loss: {:.6}".format(total_loss /
                                                             len(y_preds)))

      else:
        predictions = {"preds": y_preds}
        self.postproc_fn()(predictions, log_verbose=False)

        if not self.infer_no_label:
          metcs = metrics.get_metrics(
              config=self.config, y_pred=y_preds, y_true=y_ground_truth)
          logging.info("Evaluation on %s:" % mode)
          # add sort function to make sequence of metrics identical.
          for key in sorted(metcs.keys()):
            logging.info(key + ":" + str(metcs[key]))
