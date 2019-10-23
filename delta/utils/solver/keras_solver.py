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

#!/usr/bin/env python
"""Keras solver is not stable now"""

import os
import math
import numpy as np
import delta.compat as tf
from absl import logging

from delta.utils.solver.base_solver import Solver

from delta import utils
from delta.utils.register import registers
from delta.utils.solver.utils.solver_utils import get_checkpoint_dir
from delta.utils.solver.utils.solver_utils import get_session_conf
from delta.utils.solver.utils.solver_utils import to_saved_model
from delta.utils.solver.utils.solver_utils import save_infer_res


@registers.solver.register
class KerasSolver(Solver):
  """Solver in Keras way."""

  def __init__(self, config):
    super().__init__(config)
    self.model_compiled = False
    self.model_path = config['solver']['saver']['model_path']
    self.checkpoint_dir = get_checkpoint_dir(self.config)
    self.session_conf = get_session_conf(self.config)
    self.session = tf.Session(config=self.session_conf)
    tf.keras.backend.set_session(self.session)
    self.metrics = self.get_metrics()

  def process_config(self, config):
    """Process configs."""
    return config

  def input_fn(self, mode):
    """Get the input function for model."""
    super().input_fn(mode)
    return self.task.input_fn()()

  def get_metrics(self):
    """Get metrics."""
    metrics_list = self.config['solver']['metrics']['keras']
    return [m["name"] for m in metrics_list]

  def build_inputs(self, mode):
    """Build the inputs."""
    inputs = self.input_fn(mode)

    self.config['data']['sequence_length'] = inputs.max_seq_len
    self.config['data']['vocab_size'] = inputs.vocab_size
    self.config['data']['{}_data_size'.format(mode)] = inputs.data_size

    return inputs

  def build(self):
    """Build the model."""

    self.model = self.model_fn()  # pylint: disable=attribute-defined-outside-init

    loss_fn = self.get_loss_fn()

    multitask = self.config['solver']['optimizer']['multitask']
    optimizer = self.get_optimizer(multitask)

    self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=self.metrics)

    self.model_compiled = True
    logging.info("Model is built.")

  def train_core(self, train_inputs, eval_inputs=None):
    """Core part of training."""

    self.build()

    self.session.run(tf.global_variables_initializer())
    self.session.run(tf.tables_initializer())
    self.session.run(train_inputs.iterator.initializer)
    if eval_inputs is not None:
      self.session.run(eval_inputs.iterator.initializer)
      validation_data = (eval_inputs.input_x_dict,
                         eval_inputs.input_y_dict["input_y"])
      eval_data_size = self.config['data']['eval_data_size']
      batch_size = self.config['data']['task']['batch_size']
      validation_steps = int(eval_data_size / batch_size)
    else:
      validation_data = None
      validation_steps = None

    train_data_size = self.config['data']['train_data_size']
    num_epochs = self.config['solver']['optimizer']['epochs']
    batch_size = self.config['data']['task']['batch_size']
    num_batch_per_epoch = int(math.ceil(train_data_size / batch_size))

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            os.path.join(self.model_path, "logs"),
            histogram_freq=0,
            write_graph=True,
            write_grads=True,
            write_images=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.checkpoint_dir, "weights.{epoch:02d}"),
            save_weights_only=True,
            save_best_only=True)
    ]

    self.model.fit(
        train_inputs.input_x_dict,
        train_inputs.input_y_dict["input_y"],
        callbacks=callbacks,
        epochs=num_epochs,
        steps_per_epoch=num_batch_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps)

  def train(self):
    """Train the model."""
    inputs = self.build_inputs(utils.TRAIN)

    self.train_core(inputs)

  def train_and_eval(self):
    """Train and evaluate the model."""
    train_inputs = self.build_inputs(utils.TRAIN)
    eval_inputs = self.build_inputs(utils.EVAL)
    self.train_core(train_inputs, eval_inputs)

  def eval(self):
    """Evaluate the model."""
    inputs = self.build_inputs(utils.EVAL)
    self.build()
    self.session.run(tf.global_variables_initializer())
    self.session.run(tf.tables_initializer())
    self.session.run(inputs.iterator.initializer)
    eval_data_size = self.config['data']['eval_data_size']
    batch_size = self.config['data']['task']['batch_size']
    steps = int(math.ceil(eval_data_size / batch_size))
    weights_ckpt_dir = tf.train.latest_checkpoint(self.checkpoint_dir)
    self.model.load_weights(weights_ckpt_dir)
    results = self.model.evaluate(
        inputs.input_x_dict, inputs.input_y_dict["input_y"], steps=steps)
    for metric, res in zip(self.model.metrics_names, results):
      print("{}: {}".format(metric, res))

  def infer(self, **kwargs):  # pylint: disable=arguments-differ, unused-argument
    """Make a inference."""
    inputs = self.build_inputs(utils.INFER)
    self.build()
    self.session.run(tf.global_variables_initializer())
    self.session.run(tf.tables_initializer())
    self.session.run(inputs.iterator.initializer)
    infer_data_size = self.config['data']['infer_data_size']
    batch_size = self.config['data']['task']['batch_size']
    steps = int(math.ceil(infer_data_size / batch_size))
    weights_ckpt_dir = tf.train.latest_checkpoint(self.checkpoint_dir)
    self.model.load_weights(weights_ckpt_dir)
    logits = self.model.predict(inputs.input_x_dict, steps=steps)
    preds = np.argmax(logits, axis=-1)
    save_infer_res(self.config, logits, preds)

  def export_model(self):
    """Export a model to tensorflow SavedModel."""
    inputs = self.build_inputs(utils.INFER)
    self.build()
    logits = self.model(inputs.input_x_dict)
    score = tf.nn.softmax(logits)

    self.session.run(tf.global_variables_initializer())
    self.session.run(tf.tables_initializer())
    self.session.run(inputs.iterator.initializer)
    weights_ckpt_dir = tf.train.latest_checkpoint(self.checkpoint_dir)
    self.model.load_weights(weights_ckpt_dir)

    output_dict = {"score": score}
    to_saved_model(self.config, self.session, inputs.input_x_dict, output_dict)
