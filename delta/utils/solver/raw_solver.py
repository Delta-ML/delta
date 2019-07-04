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
"""Solver for raw tensorflow model."""

import math
import numpy as np
import tensorflow as tf
from absl import logging

from delta.utils.solver.base_solver import Solver

from delta import utils
from delta.utils import metrics
from delta.utils.register import registers
from delta.utils.solver.solver_utils import get_checkpoint_dir
from delta.utils.solver.solver_utils import get_ckpt_state
from delta.utils.solver.solver_utils import get_session_conf
from delta.utils.solver.solver_utils import to_saved_model

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation


@registers.solver.register
class RawSolver(Solver):
  """Solver for raw tensorflow model."""

  def __init__(self, config):
    super().__init__(config)
    self.session_conf, self.smax_to_keep, self.batch_size, self.num_epochs, \
    self.save_checkpoint_steps, \
    self.resume_model_path, self.print_every = self.set_experimental_environment()
    self.first_eval = True
    self.do_eval = False
    self.infer_no_label = self.config['data'][utils.INFER].get(
        'infer_no_label', False)

  def process_config(self, config):
    """Process the configs."""
    return config

  def input_fn(self, mode):
    """Get the input function for training, evaluation and inference."""
    super().input_fn(mode)
    return self.task.input_fn()()

  def export_input(self, mode):
    """Get the input function for model export."""
    super().input_fn(mode)
    return self.task.export_inputs()

  def set_experimental_environment(self):
    """Set the experimental environment."""
    # Set configuration
    session_conf = get_session_conf(self.config)

    task_config = self.config["data"]["task"]
    batch_size = task_config['batch_size']
    num_epochs = task_config['epochs']

    saver_conf = self.config['solver']['saver']
    smax_to_keep = saver_conf['max_to_keep']
    save_checkpoint_steps = saver_conf['save_checkpoint_steps']
    resume_model_path = saver_conf.get('resume_model_path', None)
    print_every = saver_conf['print_every']

    return session_conf, smax_to_keep, batch_size, num_epochs, \
        save_checkpoint_steps, \
        resume_model_path, print_every

  def get_scaffold(self, global_step=None, iter_initializer=None):  #pylint: disable=arguments-differ
    """Get training scaffold."""
    init_op = tf.global_variables_initializer()
    if iter_initializer is None:
      local_init_op = tf.tables_initializer()
    else:
      local_init_op = tf.group(tf.tables_initializer(), iter_initializer)
    saver = self.get_saver(global_step)
    scaffold = tf.train.Scaffold(
        saver=saver, init_op=init_op, local_init_op=local_init_op)
    return scaffold

  def get_generated_model_path(self):
    """Get the path of the checkpoint which is most recently generated during training process."""
    ckpt = get_ckpt_state(self.config)
    if ckpt is None:
      return None
    model_path = ckpt.model_checkpoint_path  # pylint: disable=no-member
    return model_path

  def get_model_path(self, mode):
    """Get the path of the checkpoint of the model."""
    model_path = ""
    if "{}_model_path".format(mode) in self.config["solver"]["saver"]:
      model_path = self.config["saver"]["{}_model_path".format(mode)]
    if model_path == "":
      model_path = self.get_generated_model_path()
    return model_path

  def build(self, mode: str):
    """Build the model for training, eval and infer."""
    inputs = self.input_fn(mode)
    logging.info("build input data done...")

    model = self.model_fn()
    training = mode == utils.TRAIN
    model.logits = model(inputs["input_x_dict"], training=training)
    model.input_x_len = inputs["input_x_len"]
    model.iterator = inputs["iterator"]
    model.input_x_dict = inputs["input_x_dict"]
    model.input_x_len = inputs["input_x_len"]
    model.loss_fn = self.get_loss_fn()
    if mode != utils.INFER or not self.infer_no_label:
      input_y = inputs["input_y_dict"]["input_y"]
      model.loss = model.loss_fn(labels=input_y, logits=model.logits,
                           input_length=model.input_x_len, name="loss")
      logging.info("model.loss done")
      model.input_y = input_y

    # output related
    self.build_output(model)
    return model

  def build_export_model(self):
    """Build the model for export."""
    mode = utils.INFER
    export_inputs = self.export_input(mode)

    model = self.model_fn()
    training = mode == utils.TRAIN
    model.logits = model(export_inputs["model_inputs"], training=training)
    model.model_inputs = export_inputs["model_inputs"]
    model.export_inputs = export_inputs["export_inputs"]
    model.loss_fn = self.get_loss_fn()
    model.input_x_len = export_inputs["model_inputs"]["input_x_len"]

    # output related
    self.build_output(model)
    return model

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """

    raise NotImplementedError

  def eval(self):
    """Evaluate the model."""
    mode = utils.EVAL
    graph = tf.Graph()
    with graph.as_default():
      self.eval_or_infer_once(mode)

  def infer(self, **kwargs):  # pylint: disable=unused-argument, arguments-differ
    """Make a inference."""
    mode = utils.INFER
    graph = tf.Graph()
    with graph.as_default():
      self.eval_or_infer_once(mode)

  def postproc_fn(self):
    """Post-process function, called after inference."""
    postproc_name = self.config['solver']['postproc']["name"]
    postproc = registers.postprocess[postproc_name](self.config)
    return postproc

  def eval_or_infer_once(self, mode):
    """Do evaluation or inference once."""
    model = self.build(mode)
    model.sess = tf.Session(config=self.session_conf)
    model.saver = tf.train.Saver()
    self.eval_or_infer_core(model, mode)
    model.sess.close()

  def eval_or_infer_core(self, model, mode):  # pylint: disable=too-many-locals, too-many-branches
    """The core part of evaluation."""

    if mode == utils.EVAL or not self.infer_no_label:
      self.do_eval = True
    else:
      self.do_eval = False
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
      y_logits = []
      y_preds = []

      logging.info("Total eval data size: {},"
                   "batch num per epoch: {}".format(data_size, num_batch_every_epoch))

      for i in range(num_batch_every_epoch):

        if self.do_eval:
          loss_val, \
          batch_logits, \
          batch_preds, \
          batch_y_ground_truth = model.sess.run(
              [model.loss, model.logits,
               model.preds, model.y_ground_truth])
        else:
          batch_logits, batch_preds = model.sess.run(
              [model.logits, model.preds])

        end_id = (i + 1) * self.batch_size

        if data_size < end_id:
          act_end_id = self.batch_size - end_id + data_size
          batch_logits = batch_logits[:act_end_id]
          batch_preds = batch_preds[:act_end_id]
          if self.do_eval:
            batch_y_ground_truth = batch_y_ground_truth[:act_end_id]

        y_logits.append(batch_logits)
        y_preds.append(batch_preds)

        if self.do_eval:
          y_ground_truth.append(batch_y_ground_truth)
          total_loss += loss_val

        if i % 10 == 0 or i == num_batch_every_epoch - 1:
          logging.info("Evaluation rate of "
                       "progress: [ {:.2%} ]".format(
                           i / (num_batch_every_epoch - 1)))

      y_logits = np.concatenate(y_logits, axis=0)
      y_preds = np.concatenate(y_preds, axis=0)
      if self.do_eval:
        y_ground_truth = np.concatenate(y_ground_truth, axis=0)

        metcs = metrics.get_metrics(
            config=self.config, y_pred=y_preds, y_true=y_ground_truth)
        logging.info("Evaluation on %s:" % mode)
        # add sort function to make sequence of metrics identical.
        for key in sorted(metcs.keys()):
          logging.info(key + ":" + str(metcs[key]))
      if mode == utils.INFER:
        predictions = {"logits": y_logits, "preds": y_preds}
        self.postproc_fn()(predictions, log_verbose=False)

  def export_model(self):
    """Export a model to tensorflow SavedModel."""
    mode = utils.INFER
    graph = tf.Graph()
    with graph.as_default():
      infer_model = self.build_export_model()
      infer_model.sess = tf.Session(config=self.session_conf)
      infer_model.saver = tf.train.Saver()

      model_path = self.get_model_path(mode)
      infer_model.saver.restore(infer_model.sess, save_path=model_path)

      to_saved_model(self.config, infer_model.sess, infer_model.export_inputs,
                     infer_model.output_dict)

  def train(self):  # pylint: disable=too-many-locals
    """Train the model."""
    mode = utils.TRAIN
    train_model = self.build(mode)

    multitask = self.config['solver']['optimizer']['multitask']

    # Supervisor
    with tf.name_scope("train"):
      global_step = tf.train.get_or_create_global_step()
      train_op = self.get_train_op(train_model.loss, multitask, global_step)

      checkpoint_dir = get_checkpoint_dir(self.config)

      # scaffold

      scaffold = self.get_scaffold(global_step,
                                   train_model.iterator.initializer)


      with tf.train.MonitoredTrainingSession(
          checkpoint_dir=checkpoint_dir,
          scaffold=scaffold,
          save_checkpoint_steps=self.save_checkpoint_steps,
          config=self.session_conf) as sess:

        # Training loop. For each batch...
        data_size = self.config['data']['train_data_size']
        num_epochs = self.config["data"]["task"]['epochs']
        num_batch = int(math.ceil(data_size * num_epochs / self.batch_size))
        num_batch_per_epoch = int(data_size / self.batch_size)

        for i in range(num_batch):
          _, _, out_loss = sess.run([train_op, global_step, train_model.loss])
          if i % self.print_every == 0 or i == num_batch - 1:
            logging.info(
                "Training for epoch {}: [ {:.2%} ] loss is {:g}".format(
                    int(i / num_batch_per_epoch),
                    (i % num_batch_per_epoch) / num_batch_per_epoch, out_loss))

  def train_and_eval(self):  # pylint: disable=too-many-locals
    """Train and evaluate the model."""
    # train related
    g_train = tf.Graph()
    with g_train.as_default():
      logging.info("Compiling train model ...")
      train_model = self.build(utils.TRAIN)
    # eval related
    g_eval = tf.Graph()
    with g_eval.as_default():
      logging.info("Compiling eval model ...")
      eval_model = self.build(utils.EVAL)
      eval_model.sess = tf.Session(config=self.session_conf, graph=g_eval)
      eval_model.saver = tf.train.Saver()

    # start train
    with g_train.as_default():
      multitask = self.config['solver']['optimizer']['multitask']

      # Supervisor
      with tf.name_scope("train"):
        global_step = tf.train.get_or_create_global_step()

        train_op = self.get_train_op(train_model.loss, multitask, global_step)

        checkpoint_dir = get_checkpoint_dir(self.config)

        # scaffold
        scaffold = self.get_scaffold(global_step,
                                     train_model.iterator.initializer)

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            scaffold=scaffold,
            save_checkpoint_steps=self.save_checkpoint_steps,
            config=self.session_conf) as sess:

          # Training loop. For each batch...
          train_data_size = self.config['data']['train_data_size']
          num_batch = int(train_data_size * self.num_epochs / self.batch_size)
          num_batch_per_epoch = int(train_data_size / self.batch_size)
          logging.info("Total data size: {}, batch num: {}, "
                       "batch num per epoch: {}".format(train_data_size,
                                                        num_batch,
                                                        num_batch_per_epoch))
          for i in range(0, num_batch):
            if i % self.save_checkpoint_steps == 0 and i != 0:
              self.eval_or_infer_core(eval_model, utils.EVAL)
            _, _, out_loss = sess.run([train_op, global_step, train_model.loss])
            if i % 10 == 0 or i == num_batch - 1 or (
                i +
                1) % num_batch_per_epoch == 0 or i % num_batch_per_epoch == 0:
              logging.info(
                  "Training for epoch {}: [ {:.2%} ] loss is {:g}".format(
                      int(i / num_batch_per_epoch),
                      (i % num_batch_per_epoch) / num_batch_per_epoch,
                      out_loss))
    eval_model.sess.close()
