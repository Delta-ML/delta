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

import re
import math
import numpy as np
import delta.compat as tf
from absl import logging

from delta.utils.solver.base_solver import Solver

from delta import utils
from delta.utils.register import registers
from delta.utils.solver.utils.solver_utils import get_checkpoint_dir
from delta.utils.solver.utils.solver_utils import get_ckpt_state
from delta.utils.solver.utils.solver_utils import get_session_conf
from delta.utils.solver.utils.solver_utils import to_saved_model
from delta.utils.solver.utils.solver_utils import run_metrics

# pylint: disable=too-many-instance-attributes, not-context-manager, bad-continuation


@registers.solver.register
class RawSolver(Solver):
  """Solver for raw tensorflow model."""

  def __init__(self, config):
    super().__init__(config)
    self.session_conf, self.smax_to_keep, \
    self.batch_size, self.num_epochs, \
    self.save_checkpoint_steps, \
    self.resume_model_path, self.print_every = self.set_experimental_environment()
    self.first_eval = True
    self.do_eval = False
    self.is_multi_output = False
    self.output_num = 1
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

  def get_scaffold(self, mode, global_step=None, iter_initializer=None):
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
      model_path = self.config["solver"]["saver"]["{}_model_path".format(mode)]
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
      if isinstance(model.loss_fn, list):
        model.loss = []
        for i, one_loss_fn in enumerate(model.loss_fn):
          one_loss = one_loss_fn(
              labels=input_y[i],
              logits=model.logits[i],
              input_length=model.input_x_len,
              model=model,
              name="loss_{}".format(i))
          model.loss.append(one_loss)
        model.loss_op = tf.add_n(model.loss, name="loss_sum")
      else:
        model.loss = model.loss_fn(
            labels=input_y,
            logits=model.logits,
            input_length=model.input_x_len,
            model=model,
            name="loss")
        model.loss_op = model.loss
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
    model.input_x_len = export_inputs["model_inputs"]["input_x_len"]

    # output related
    self.build_export_output(model)
    return model

  def build_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model.
    `score` and `input_y` are for loss calculation.
    `preds` and `y_ground_truth` are for metric calculation.
    """
    raise NotImplementedError

  def build_export_output(self, model):  # pylint: disable=no-self-use
    """
    Build the output of the model for export.
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
    postproc = self.config['solver']['postproc']
    if isinstance(postproc, list):
      postproc_fn = []
      for one_postproc in postproc:
        postproc_fn.append(registers.postprocess[one_postproc["name"]](
            self.config))
    else:
      postproc_fn = registers.postprocess[postproc["name"]](self.config)
    return postproc_fn

  def eval_or_infer_once(self, mode):
    """Do evaluation or inference once."""
    model = self.build(mode)
    model.sess = tf.Session(config=self.session_conf)
    model.saver = tf.train.Saver()
    self.eval_or_infer_core(model, mode)
    model.sess.close()

  def eval_or_infer_core(self, model, mode):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """The core part of evaluation."""

    self.do_eval = bool(mode == utils.EVAL or not self.infer_no_label)
    self.is_multi_output = bool(isinstance(model.preds, (tuple, list)))
    if self.is_multi_output:
      self.output_num = len(model.preds)
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
      data_size = self.config["data"]['{}_data_size'.format(mode)]
      num_batch_every_epoch = int(math.ceil(data_size / self.batch_size))

      all_fetch_vals = []

      logging.info("Total eval data size: {},"
                   "batch num per epoch: {}".format(data_size,
                                                    num_batch_every_epoch))

      for i in range(num_batch_every_epoch):
        if self.do_eval:
          if self.is_multi_output:
            fetch_ops = model.loss + list(model.logits) + list(
                model.preds) + list(model.y_ground_truth)
          else:
            fetch_ops = [
                model.loss, model.logits, model.preds, model.y_ground_truth
            ]
        else:
          fetch_ops = [model.logits, model.preds]
        logging.debug("fetch_ops: {}".format(fetch_ops))
        fetch_vals = model.sess.run(fetch_ops)

        end_id = (i + 1) * self.batch_size

        if data_size < end_id:
          logging.debug("data_size: {}, end_id: {}".format(data_size, end_id))
          act_end_id = self.batch_size - end_id + data_size
          new_fetch_vals = []
          for fetch_val in fetch_vals:
            if np.isscalar(fetch_val):
              new_fetch_vals.append(fetch_val)
            else:
              new_fetch_vals.append(fetch_val[:act_end_id])
        else:
          new_fetch_vals = fetch_vals

        all_fetch_vals.append(new_fetch_vals)

        if i % self.print_every == 0 or i == num_batch_every_epoch - 1:
          logging.info("Evaluation rate of "
                       "progress: [ {:.2%} ]".format(
                           i / (num_batch_every_epoch - 1)))

      all_fetch_nps = []
      for one_fetch_vals in zip(*all_fetch_vals):
        if len(np.shape(one_fetch_vals[0])) <= 0:  # pylint: disable=len-as-condition
          one_fetch_np = one_fetch_vals
        else:
          one_fetch_np = np.concatenate(one_fetch_vals, axis=0)
        all_fetch_nps.append(one_fetch_np)

      # reshape for multi-output
      if self.is_multi_output:
        logging.debug("all_fetch_nps before reshape: {}".format(
            len(all_fetch_nps)))
        new_all_fetch_nps = []
        sub_fetch_nps = []
        for one_fetch_np in all_fetch_nps:
          sub_fetch_nps.append(one_fetch_np)
          if len(sub_fetch_nps) == self.output_num:
            new_all_fetch_nps.append(sub_fetch_nps)
            sub_fetch_nps = []

        logging.debug("new_all_fetch_nps after reshape: {}".format(
            len(new_all_fetch_nps)))
      else:
        new_all_fetch_nps = all_fetch_nps

      if self.do_eval:
        _, _, preds_val, y_ground_truth_val = new_all_fetch_nps
        run_metrics(self.config, preds_val, y_ground_truth_val, mode)

      if mode == utils.INFER:
        if self.do_eval:
          _, logits_val, preds_val, _ = new_all_fetch_nps
        else:
          logits_val, preds_val = new_all_fetch_nps

        postproc_fn = self.postproc_fn()
        logging.info(postproc_fn)
        if isinstance(postproc_fn, list):
          for i, one_postproc_fn in enumerate(postproc_fn):
            predictions = {
                "logits": logits_val[i],
                "preds": preds_val[i],
                "output_index": i
            }
            one_postproc_fn(predictions, log_verbose=False)
        else:
          predictions = {
              "logits": logits_val,
              "preds": preds_val,
              "output_index": None
          }
          postproc_fn(predictions, log_verbose=False)

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
      train_op = self.get_train_op(train_model.loss_op, multitask, global_step)

      checkpoint_dir = get_checkpoint_dir(self.config)

      # scaffold
      scaffold = self.get_scaffold(mode, global_step, train_model.iterator.initializer)

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
      logging.info(
          "num_batch: {}, num_batch_per_epoch: {}, num_epochs: {}".format(
              num_batch, num_batch_per_epoch, num_epochs))
      for i in range(num_batch):
        _, _, out_loss = sess.run([train_op, global_step, train_model.loss_op])
        if i % self.print_every == 0 or i == num_batch - 1:
          logging.info("Training for epoch {}: [ {:.2%} ] loss is {:g}".format(
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

        train_op = self.get_train_op(train_model.loss_op, multitask,
                                     global_step)

        checkpoint_dir = get_checkpoint_dir(self.config)

        # scaffold
        scaffold = self.get_scaffold(utils.TRAIN, global_step, train_model.iterator.initializer)

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            scaffold=scaffold,
            save_checkpoint_steps=self.save_checkpoint_steps,
            config=self.session_conf) as sess:
          # Training loop. For each batch...
          train_data_size = self.config['data']['train_data_size']
          num_batch = math.ceil(train_data_size * self.num_epochs /
                                self.batch_size)
          num_batch_per_epoch = math.ceil(train_data_size / self.batch_size)
          logging.info("Total data size: {}, batch num: {}, "
                       "batch num per epoch: {}".format(train_data_size,
                                                        num_batch,
                                                        num_batch_per_epoch))
          for i in range(0, num_batch):

            if i % self.save_checkpoint_steps == 0 and i != 0:
              self.eval_or_infer_core(eval_model, utils.EVAL)
            _, _, out_loss = sess.run(
                [train_op, global_step, train_model.loss_op])
            if i % self.print_every == 0 or i == num_batch - 1 or (
                i +
                1) % num_batch_per_epoch == 0 or i % num_batch_per_epoch == 0:
              logging.info(
                  "Training for epoch {}: [ {:.2%} ] loss is {:g}".format(
                      int(i / num_batch_per_epoch),
                      (i % num_batch_per_epoch) / num_batch_per_epoch,
                      out_loss))
    eval_model.sess.close()
