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
"""Base classes of solver."""

import abc
import math
import delta.compat as tf
from absl import logging

from delta import utils
from delta.utils import optimizer
from delta.utils.register import registers

# pylint: disable=abstract-method


class ABCSolver(metaclass=abc.ABCMeta):
  """Abstract class of solver."""

  @abc.abstractmethod
  def process_config(self, config):
    """Process the configs."""
    raise NotImplementedError()

  @abc.abstractmethod
  def input_fn(self, mode):
    """Get the input function."""
    raise NotImplementedError()

  @abc.abstractmethod
  def model_fn(self):
    """Get the model function."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_loss_fn(self):
    """Get the loss function."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_learning_rate(self):
    """Get the learning rate."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_optimizer(self):
    """Get the optimizer."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_apply_gradients_op(self):
    """Get the apply gradients operator."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_train_op(self):
    """Get the training operator."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_saver(self):
    """Get the saver."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_scaffold(self):
    """Get the scaffold."""
    raise NotImplementedError()

  @abc.abstractmethod
  def train(self):
    """Train the model."""
    raise NotImplementedError()

  @abc.abstractmethod
  def eval(self):
    """Evaluate the model."""
    raise NotImplementedError()

  @abc.abstractmethod
  def infer(self):
    """Make a inference."""
    raise NotImplementedError()

  @abc.abstractmethod
  def train_and_eval(self):
    """Train and evaluate."""
    raise NotImplementedError()

  @abc.abstractmethod
  def export_model(self):
    """Export model to tensorflow SavedModel."""
    raise NotImplementedError()


class Solver(ABCSolver):
  """Base class of solver."""

  def __init__(self, config):
    super().__init__()
    self._config = self.process_config(config)
    self._task = None

  @property
  def config(self):
    """Get the config."""
    return self._config

  def input_fn(self, mode):
    """Get the input function.
    return a Task class
    """
    task_name = self.config['data']['task']["name"]
    self._task = registers.task[task_name](self.config, mode)
    return self._task

  @property
  def task(self):
    """Get the task."""
    return self._task

  def model_fn(self):
    ''' return Model class '''
    classname = self.config['model']['name']
    logging.info("__name__=%s\tclassname==%s", __name__, classname)

    # Model initialization
    model = registers.model[classname](self.config)
    return model

  def get_loss_fn(self):
    """Get the loss function."""
    return utils.misc.losses(self.config)

  def get_learning_rate(self):
    """Get the learning rate."""
    lrconf = self.config['solver']['optimizer']['learning_rate']
    learning_rate = lrconf['rate']
    learning_type = lrconf['type']

    #pylint: disable=invalid-name
    if learning_type == 'exp_decay':
      lr = tf.train.exponential_decay(
          learning_rate,
          tf.train.get_or_create_global_step(),
          lrconf['decay_steps'],
          lrconf['decay_rate'],
          staircase=True)
    elif learning_type == 'piecewise':
      #boundaries = [15000, 30000]
      #values = [1e-3, 1e-4, 1e-5]
      boundaries = lrconf['boundaries']
      values = lrconf['values']
      assert len(values) == len(
          boundaries) + 1, 'values len must equal boundaries len plus one'
      lr = tf.train.piecewise_constant(
          tf.train.get_or_create_global_step(),
          boundaries=boundaries,
          values=values)
    elif learning_type == 'warmup':
      learning_rate = tf.constant(
          value=learning_rate, shape=[], dtype=tf.float32)
      global_step = tf.train.get_or_create_global_step()
      data_size = self.config['data']['train_data_size']
      num_epochs = self.config["data"]["task"]['epochs']
      batch_size = self.config["data"]["task"]['batch_size']
      num_batch = int(math.ceil(data_size * num_epochs / batch_size))
      learning_rate = tf.train.polynomial_decay(
          learning_rate,
          global_step,
          num_batch,
          end_learning_rate=0.0,
          power=1.0,
          cycle=False)
      global_steps_int = tf.cast(global_step, tf.int32)
      warmup_steps_int = tf.constant(lrconf['num_warmup_steps'], dtype=tf.int32)

      global_steps_float = tf.cast(global_steps_int, tf.float32)
      warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

      warmup_percent_done = global_steps_float / warmup_steps_float
      warmup_learning_rate = learning_rate * warmup_percent_done

      is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
      lr = ((1.0 - is_warmup) * learning_rate +
            is_warmup * warmup_learning_rate)
    elif learning_type == 'const':
      lr = learning_rate
    else:
      raise ValueError(
          "Not support learning rate type: {}".format(learning_type))
    tf.summary.scalar('lr', lr)
    return lr

  #pylint: disable=arguments-differ
  def get_optimizer(self, multitask):
    """Get the optimizer."""
    optconf = self.config['solver']['optimizer']
    method = optconf['name']
    learning_rate = self.get_learning_rate()
    if method == 'adadelta':
      opt = tf.train.AdadeltaOptimizer(learning_rate)
    elif method == 'adam':
      opt = tf.train.AdamOptimizer(learning_rate)
    elif method == 'adagrad':
      opt = tf.train.AdagradOptimizer(learning_rate)
    elif method == 'momentum':
      opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif method == 'rmsprop':
      opt = tf.train.RMSPropOptimizer(learning_rate)
    elif method == 'gradientdecent':
      opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif method == 'lazyadam':
      opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate)
    elif method == 'weightedadam':
      weight_decay = self.config['solver']['optimizer']['weight_decay']
      opt = tf.contrib.opt.AdamWOptimizer(
          weight_decay=weight_decay, learning_rate=learning_rate)
    elif method == 'yellowfin':
      opt = optimizer.YFOptimizer(learning_rate)
    else:
      raise ValueError("Not support optimizer: {}".format(method))

    if multitask:
      opt = tf.contrib.opt.MultitaskOptimizerWrapper(opt)
      logging.info("Using multi-task optimizer")
    return opt

  #pylint: disable=no-self-use
  def clip_gradients(self, grads_and_vars, clip_ratio, multitask=False):
    """Clip the gradients."""
    is_zip_obj = False
    if isinstance(grads_and_vars, zip):
      grads_and_vars = list(grads_and_vars)
      is_zip_obj = True

    with tf.variable_scope('grad'):
      for grad, var in grads_and_vars:
        if grad is not None:
          if tf.executing_eagerly():
            tf.contrib.summary.histogram(var.name[:-2], grad)
          else:
            tf.summary.histogram(var.name[:-2], grad)
        else:
          logging.debug('%s gradient is None' % (var.name))

    # not clip
    if not clip_ratio:
      if is_zip_obj:
        grads, variables = zip(*grads_and_vars)
        grads_and_vars = zip(grads, variables)
      return grads_and_vars

    if multitask:
      grad_and_var_clipped, global_norm = tf.contrib.opt.clip_gradients_by_global_norm(
          grads_and_vars, clip_ratio)
    else:
      gradients, variables = zip(*grads_and_vars)
      clipped, global_norm = tf.clip_by_global_norm(gradients, clip_ratio)
      grad_and_var_clipped = zip(clipped, variables)

    if tf.executing_eagerly():
      tf.contrib.summary.scalar('gradient/global_norm', global_norm)
    else:
      tf.summary.scalar('gradient/global_norm', global_norm)

    return grad_and_var_clipped

  def get_apply_gradients_op(self, loss, multitask, global_step=None):
    """Get Apply gradients operator."""
    opt = self.get_optimizer(multitask)
    grads_and_vars = opt.compute_gradients(loss)

    # clip gradient
    optconf = self.config['solver']['optimizer']
    global_norm = optconf['clip_global_norm']
    grads_and_vars = self.clip_gradients(grads_and_vars, global_norm, multitask)

    apply_gradient_op = opt.apply_gradients(
        grads_and_vars,
        global_step=global_step or tf.train.get_or_create_global_step())
    return apply_gradient_op

  def get_var_avg_ema(self, decay, global_step=None):
    ''' make var average ema '''
    return tf.train.ExponentialMovingAverage(
        decay, global_step or tf.train.get_or_create_global_step())

  def make_restore_average_vars_dict(self, global_step=None):
    ''' using vars_average to restotre vars'''
    model_avg_conf = self.config['solver']['model_average']
    var_avg_decay = model_avg_conf['var_avg_decay']

    var_restore_dict = {}
    variable_averages = self.get_var_avg_ema(var_avg_decay, global_step)
    for var in tf.global_variables():
      if var in tf.trainable_variables():
        name = variable_averages.average_name(var)
      else:
        name = var.op.name
      var_restore_dict[name] = var
    return var_restore_dict

  def var_avg(self, global_step=None):
    ''' average model variables, add average_op to UPDATES_OPS'''
    model_avg_conf = self.config['solver']['model_average']
    var_avg_model = model_avg_conf['enable']
    if var_avg_model:
      var_avg_decay = model_avg_conf['var_avg_decay']
      variable_averages = self.get_var_avg_ema(var_avg_decay, global_step)
      apply_op = variable_averages.apply(tf.trainable_variables())
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_op)
      utils.log_vars('Avg Trainable Vars', tf.trainable_variables())

  def get_train_op(self, loss, multitask, global_step=None):
    """Get the training operator."""
    # quantize training
    quantconf = self.config['solver']['quantization']
    quantization = quantconf['enable']
    if quantization:
      quant_delay = quantconf['quant_delay']
      logging.info('Quantization training with {} delay'.format(quant_delay))
      tf.contrib.quantize.create_training_graph(quant_delay=quant_delay)

    apply_gradient_op = self.get_apply_gradients_op(loss, multitask,
                                                    global_step)

    # model average
    self.var_avg(global_step)

    # model average after apply gradients
    with tf.control_dependencies([apply_gradient_op]):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      train_op = tf.group(*update_ops)

    utils.log_vars('moving vars', tf.moving_average_variables())
    return train_op

  def get_saver(self, global_step=None):
    """Get the saver."""
    solverconf = self.config['solver']
    max_to_keep = solverconf['saver']['max_to_keep']
    model_avg_conf = self.config['solver']['model_average']
    model_average = model_avg_conf['enable']
    if model_average:
      var_avg_decay = model_avg_conf['var_avg_decay']
      variable_averages = self.get_var_avg_ema(var_avg_decay, global_step)
      variable_to_restore = variable_averages.variables_to_restore()
      logging.info('Restore: name to var : {}'.format(variable_to_restore))
      saver = tf.train.Saver(variable_to_restore, max_to_keep=max_to_keep)
      logging.info('Restore vars from moving variables')
    else:
      saver = tf.train.Saver(max_to_keep=max_to_keep)
    return saver

  def get_scaffold(self, mode, global_step=None):
    """Get the scaffold."""
    if mode != utils.TRAIN:
      # for model average
      saver = self.get_saver(global_step)
      scaffold = tf.train.Scaffold(saver=saver)
    else:
      scaffold = None  # default
    return scaffold


class ABCEstimatorSolver(Solver):
  """Abstract solver using tensorflow Esitimator."""

  @abc.abstractmethod
  def create_estimator(self):
    ''' create tf.estimator.Estimator obj'''
    raise NotImplementedError()

  @abc.abstractmethod
  def get_train_hooks(self, labels, logits, alpha=None):
    ''' return train_hooks '''
    raise NotImplementedError()

  @abc.abstractmethod
  def get_eval_hooks(self, labels, logits):
    ''' return eval_hooks, eval_metric_ops '''
    raise NotImplementedError()

  @abc.abstractmethod
  def get_infer_predictions(self):
    ''' get infer predictions output'''
    raise NotImplementedError()

  @abc.abstractmethod
  def create_serving_input_receiver_fn(self):
    ''' input pipeline when export model '''
    raise NotImplementedError()

  @abc.abstractmethod
  def postproc_fn(self):
    ''' postprocess of predictions'''
    raise NotImplementedError()
