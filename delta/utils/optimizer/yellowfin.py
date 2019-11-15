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
"""
YellowFin optimizer.

YellowFin and the Art of Momentum Tuning
https://arxiv.org/abs/1706.03471

repo: https://github.com/JianGoForIt/YellowFin
license: Apache-2.0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import delta.compat as tf
from tensorflow.python.framework import ops

# EPS for numerical stability
EPS = 1e-6
LARGE_FLOAT_VAL = 1e15


class YFOptimizer(object):
  """
  Optimizer that implements the YellowFin algorithm.

  Implemented as a wrapper around tf.train.MomentumOptimizer
  """
  # Available gate_gradients values
  GATE_NONE = tf.train.Optimizer.GATE_NONE
  GATE_OP = tf.train.Optimizer.GATE_OP
  GATE_GRAPH = tf.train.Optimizer.GATE_GRAPH

  def __init__(self,
               learning_rate=0.0001,
               momentum=0.0,
               clip_thresh=None,
               beta=0.999,
               curv_win_width=20,
               zero_debias=True,
               delta_mu=0.0,
               sparsity_debias=False,
               use_locking=False,
               name="YellowFin",
               use_nesterov=False,
               use_unsmoothed_lr_mu=True,
               h_max_log_smooth=True,
               h_min_log_smooth=True,
               use_adapt_grad_clip=True,
               stat_protect_fac=100.0):
    """
    Construct a new YellowFin optimizer.

    Args:
      learning rate: Python scalar. The initial value of learning rate,
        we use 1.0 in our paper.
      momentum: Python scalar. The initial value of momentum, we use
        0.0 in our paper.
      clip_thresh: Python scalar. The cliping threshold for
        `tf.clip_by_global_norm`. If None, no clipping will be used.
      beta: Python scalar. The smoothing parameter for estimations.
      curv_win_width: TODO
      zero_debias: TODO
      delta_mu: for extensions. Not necessary in the basic use.
      sparsity_debias: Python boolean. Gradient norm and curvature are
        biased to larger values when calculated with sparse gradient.
        This is useful when the model is very sparse, e.g. LSTM with
        word embedding. For non-sparse CNN, turning it off could
        slightly accelerate the speed.
      use_locking: If True, use locks for update operations.
      name: Optional name prefix for the operations created when
        applying gradients. Defaults to "YellowFin".
      use_nesterov: If True, the underlying MomentumOptimizer uses Nesterov
        Momentum. Set to False in the default YellowFin algorithm.

    Notes:
      `clip_thresh` is the threshold value on ||lr * gradient||
      `delta_mu` can be a placeholder/variable/python scalar. Used for
      additional momentum in situations such as asynchronous-parallel
      training. The default is 0.0 for basic usage of the optimizer.

    Other features:
      If you want to manually control the learning rates,
      `self.lr_factor` is an interface to the outside. It is a
      multiplier for the internal learning rate in YellowFin. It is
      helpful when you want to do additional hand tuning or some
      decaying scheme for the internal learning rate. Example on using
      `lr_factor` can be found here:
      https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L140
    """
    self._lr = learning_rate
    self._mu = momentum

    self._lr_var = tf.Variable(
        learning_rate, dtype=tf.float32, name="YF_lr", trainable=False)
    self._mu_var = tf.Variable(
        momentum, dtype=tf.float32, name="YF_mu", trainable=False)
    # for step scheme or decaying scheme for the learning rates
    self.lr_factor = tf.Variable(
        1.0, dtype=tf.float32, name="YF_lr_factor", trainable=False)
    if clip_thresh is not None:
      self._clip_thresh_var = tf.Variable(
          clip_thresh, dtype=tf.float32, name="YF_clip_thresh", trainable=False)
    else:
      self._clip_thresh_var = None

    # the underlying momentum optimizer
    self._optimizer = tf.train.MomentumOptimizer(self._lr_var * self.lr_factor,
                                                 self._mu_var + delta_mu,
                                                 use_locking, name,
                                                 use_nesterov)

    # moving average for statistics
    self._beta = beta
    self._moving_averager = None

    # for global step counting
    self._global_step = tf.Variable(0, trainable=False)

    self._do_tune = tf.greater(self._global_step, tf.constant(0))

    self._zero_debias = zero_debias
    self._sparsity_debias = sparsity_debias

    self._tvars = None

    # for curvature range
    self._curv_win_width = curv_win_width
    self._curv_win = None

    # option for using smoothed or unsmoothed lr and mu
    self._use_unsmoothed_lr_mu = use_unsmoothed_lr_mu

    # options for curvature envelop smoothing
    self._h_max_log_smooth = h_max_log_smooth
    self._h_min_log_smooth = h_min_log_smooth

    # for adaptive gradient clipping
    self._use_adapt_grad_clip = use_adapt_grad_clip
    self._adapt_grad_clip_thresh = \
      tf.Variable(LARGE_FLOAT_VAL, dtype=tf.float32, trainable=False)
    self._adapt_grad_clip_target_val = \
      tf.Variable(LARGE_FLOAT_VAL, dtype=tf.float32, trainable=False)

    # prevent exploding gradient from ruining the statistics
    self._stat_protect_fac = stat_protect_fac

  def curvature_range(self):
    # set up the curvature window
    self._curv_win = tf.Variable(
        np.zeros([
            self._curv_win_width,
        ]),
        dtype=tf.float32,
        name="curv_win",
        trainable=False)
    # we can use log smoothing for curvature range to follow trend faster
    # self._curv_win = tf.scatter_update(
    #   self._curv_win, self._global_step % self._curv_win_width,
    #   tf.log(self._grad_norm_squared + EPS))
    self._curv_win = tf.scatter_update(self._curv_win,
                                       self._global_step % self._curv_win_width,
                                       self._grad_norm_squared + EPS)
    # note here the iterations start from iteration 0
    valid_window = tf.slice(
        self._curv_win, tf.constant([
            0,
        ]),
        tf.expand_dims(
            tf.minimum(
                tf.constant(self._curv_win_width), self._global_step + 1),
            dim=0))

    if self._h_min_log_smooth:
      self._h_min_t = tf.log(tf.reduce_min(valid_window) + EPS)
    else:
      self._h_min_t = tf.reduce_min(valid_window)
    if self._h_max_log_smooth:
      self._h_max_t = tf.log(tf.reduce_max(valid_window) + EPS)
    else:
      self._h_max_t = tf.reduce_max(valid_window)

    curv_range_ops = []
    with tf.control_dependencies([self._h_min_t, self._h_max_t]):
      avg_op = self._moving_averager.apply([self._h_min_t, self._h_max_t])
      with tf.control_dependencies([avg_op]):
        if self._h_min_log_smooth:
          self._h_min = tf.exp(
              tf.identity(self._moving_averager.average(self._h_min_t)))
        else:
          self._h_min = \
            tf.identity(self._moving_averager.average(self._h_min_t))
        if self._h_max_log_smooth:
          self._h_max = tf.exp(
              tf.identity(self._moving_averager.average(self._h_max_t)))
        else:
          self._h_max = \
            tf.identity(self._moving_averager.average(self._h_max_t))
      if self._sparsity_debias:
        self._h_min = self._h_min * self._sparsity_avg
        self._h_max = self._h_max * self._sparsity_avg
    curv_range_ops.append(avg_op)
    return curv_range_ops

  def grad_variance(self):
    grad_var_ops = []
    tensor_to_avg = []
    for t, g in zip(self._tvars, self._grads):
      if isinstance(g, ops.IndexedSlices):
        tensor_to_avg.append(
            tf.reshape(
                tf.unsorted_segment_sum(g.values, g.indices, g.dense_shape[0]),
                shape=t.get_shape()))
      else:
        tensor_to_avg.append(g)
    avg_op = self._moving_averager.apply(tensor_to_avg)
    grad_var_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._grad_avg = [
          self._moving_averager.average(val) for val in tensor_to_avg
      ]
      self._grad_avg_squared = [tf.square(val) for val in self._grad_avg]
    self._grad_var = tf.maximum(
        tf.constant(EPS, dtype=self._grad_norm_squared_avg.dtype),
        self._grad_norm_squared_avg -
        tf.add_n([tf.reduce_sum(val) for val in self._grad_avg_squared]))
    if self._sparsity_debias:
      self._grad_var *= self._sparsity_avg
    return grad_var_ops

  def dist_to_opt(self):
    dist_to_opt_ops = []
    # running average of the norm of gradeint
    self._grad_norm = tf.sqrt(self._grad_norm_squared)
    avg_op = self._moving_averager.apply([
        self._grad_norm,
    ])
    dist_to_opt_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._grad_norm_avg = self._moving_averager.average(self._grad_norm)
      # single iteration distance estimation
      # note that self._grad_norm_avg is per variable
      self._dist_to_opt = (
          self._grad_norm_avg / (self._grad_norm_squared_avg + EPS))
    # running average of distance
    avg_op = self._moving_averager.apply([self._dist_to_opt])
    dist_to_opt_ops.append(avg_op)
    with tf.control_dependencies([avg_op]):
      self._dist_to_opt_avg = tf.identity(
          self._moving_averager.average(self._dist_to_opt))
      if self._sparsity_debias:
        self._dist_to_opt_avg /= (tf.sqrt(self._sparsity_avg) + EPS)
    return dist_to_opt_ops

  def grad_sparsity(self):
    # If the sparse minibatch gradient has 10 percent of its entries
    # non-zero, its sparsity is 0.1.
    # The norm of dense gradient averaged from full dataset
    # are roughly estimated norm of minibatch
    # sparse gradient norm * sqrt(sparsity)
    # An extension maybe only correct the sparse blob.
    non_zero_cnt = tf.add_n([tf.count_nonzero(g) for g in self._grads])
    all_entry_cnt = tf.add_n([tf.size(g) for g in self._grads])
    self._sparsity = tf.cast(non_zero_cnt, self._grads[0].dtype) \
      / tf.cast(all_entry_cnt, self._grads[0].dtype)
    avg_op = self._moving_averager.apply([
        self._sparsity,
    ])
    with tf.control_dependencies([avg_op]):
      self._sparsity_avg = self._moving_averager.average(self._sparsity)
    return avg_op

  def before_apply(self):
    self._moving_averager = tf.train.ExponentialMovingAverage(
        decay=self._beta, zero_debias=self._zero_debias)
    assert self._grads is not None and len(self._grads) > 0
    before_apply_ops = []

    # get per var g**2 and norm**2
    self._grad_squared = []
    self._grad_norm_squared = []
    for v, g in zip(self._tvars, self._grads):
      if g is None:
        continue
      with ops.colocate_with(v):
        self._grad_squared.append(tf.square(g))
    self._grad_norm_squared = [
        tf.reduce_sum(grad_squared) for grad_squared in self._grad_squared
    ]

    if self._sparsity_debias:
      avg_op_sparsity = self.grad_sparsity()
      before_apply_ops.append(avg_op_sparsity)

    # the following running average on squared norm of gradient is shared
    # by `grad_variance` and `dist_to_opt`
    avg_op = self._moving_averager.apply(self._grad_norm_squared)
    with tf.control_dependencies([avg_op]):
      self._grad_norm_squared_avg = [
          self._moving_averager.average(val) for val in self._grad_norm_squared
      ]
      self._grad_norm_squared = tf.add_n(self._grad_norm_squared)
      self._grad_norm_squared_avg = tf.add_n(self._grad_norm_squared_avg)
    before_apply_ops.append(avg_op)

    with tf.control_dependencies([avg_op]):
      curv_range_ops = self.curvature_range()
      before_apply_ops += curv_range_ops
      grad_var_ops = self.grad_variance()
      before_apply_ops += grad_var_ops
      dist_to_opt_ops = self.dist_to_opt()
      before_apply_ops += dist_to_opt_ops
    return tf.group(*before_apply_ops)

  def get_lr_tensor(self):
    lr = (1.0 - tf.sqrt(self._mu))**2 / (self._h_min + EPS)
    lr = tf.minimum(
        lr,
        lr * (tf.to_float(self._global_step) + 1.0) / 10.0 /
        tf.to_float(tf.constant(self._curv_win_width)))
    return lr

  def get_cubic_root(self):
    # We have the equation x^2 D^2 + (1-x)^4 * C / h_min^2
    # where x = sqrt(mu).
    # We substitute x, which is sqrt(mu), with x = y + 1.
    # It gives y^3 + py = q
    # where p = (D^2 h_min^2)/(2*C) and q = -p.
    # We use the Vieta's substution to compute the root.
    # There is only one real solution y (which is in [0, 1] ).
    # http://mathworld.wolfram.com/VietasSubstitution.html
    # assert_array = \
    #   [tf.Assert(tf.logical_not(tf.is_nan(self._dist_to_opt_avg) ), [self._dist_to_opt_avg,]),
    #   tf.Assert(tf.logical_not(tf.is_nan(self._h_min) ), [self._h_min,]),
    #   tf.Assert(tf.logical_not(tf.is_nan(self._grad_var) ), [self._grad_var,]),
    #   tf.Assert(tf.logical_not(tf.is_inf(self._dist_to_opt_avg) ), [self._dist_to_opt_avg,]),
    #   tf.Assert(tf.logical_not(tf.is_inf(self._h_min) ), [self._h_min,]),
    #   tf.Assert(tf.logical_not(tf.is_inf(self._grad_var) ), [self._grad_var,])]
    # with tf.control_dependencies(assert_array):
    # EPS in the numerator to prevent momentum being exactly one in case of 0 gradient
    p = (self._dist_to_opt_avg + EPS)**2 * (self._h_min + EPS)**2 / 2 / (
        self._grad_var + EPS)
    w3 = (-tf.sqrt(p**2 + 4.0 / 27.0 * p**3) - p) / 2.0
    w = tf.sign(w3) * tf.pow(tf.abs(w3), 1.0 / 3.0)
    y = w - p / 3.0 / (w + EPS)
    x = y + 1
    return x

  def get_mu_tensor(self):
    root = self.get_cubic_root()
    dr = tf.maximum((self._h_max + EPS) / (self._h_min + EPS), 1.0 + EPS)
    mu = tf.maximum(root**2, ((tf.sqrt(dr) - 1) / (tf.sqrt(dr) + 1))**2)
    return mu

  def update_hyper_param(self):
    assign_hyper_ops = []
    self._mu = tf.identity(
        tf.cond(self._do_tune, lambda: self.get_mu_tensor(),
                lambda: self._mu_var))
    with tf.control_dependencies([self._mu]):
      self._lr = tf.identity(
          tf.cond(self._do_tune, lambda: self.get_lr_tensor(),
                  lambda: self._lr_var))

    with tf.control_dependencies([self._mu, self._lr]):
      if self._use_unsmoothed_lr_mu:
        assign_hyper_ops.append(tf.assign(self._mu_var, self._mu))
        assign_hyper_ops.append(tf.assign(self._lr_var, self._lr))
      else:
        self._mu = self._beta * self._mu_var + (1 - self._beta) * self._mu
        self._lr = self._beta * self._lr_var + (1 - self._beta) * self._lr
        with tf.control_dependencies([self._mu, self._lr]):
          assign_hyper_ops.append(tf.assign(self._mu_var, self._mu))
          assign_hyper_ops.append(tf.assign(self._lr_var, self._lr))
    assign_hyper_op = tf.group(*assign_hyper_ops)
    return assign_hyper_op

  def get_name(self):
    return self._optimizer.get_name()

  def apply_gradients(self, grads_tvars, global_step=None, name=None):
    self._grads, self._tvars = zip(
        *[(g, t) for g, t in grads_tvars if g is not None])

    # for manual gradient clipping
    if self._clip_thresh_var is not None:
      self._grads, self._grads_norm = tf.clip_by_global_norm(
          self._grads, self._clip_thresh_var)

    # loosely adaptive clipping of gradient in case exploding gradient ruins statistics
    if self._use_adapt_grad_clip:
      thresh = tf.cond(
          self._do_tune, lambda: tf.sqrt(self._stat_protect_fac * self.
                                         _adapt_grad_clip_thresh**2),
          lambda: tf.to_float(tf.constant(LARGE_FLOAT_VAL)))
      self._grads, self._grads_norm = tf.clip_by_global_norm(
          self._grads, thresh)

    with tf.variable_scope("before_apply"):
      before_apply_op = self.before_apply()

    with tf.variable_scope("update_hyper"):
      with tf.control_dependencies([before_apply_op]):
        update_hyper_op = self.update_hyper_param()

    with tf.variable_scope("apply_updates"):
      with tf.control_dependencies([update_hyper_op]):

        # clip exploding gradient according to h_max
        if self._use_adapt_grad_clip:
          thresh = tf.cond(
              tf.greater(
                  tf.global_norm(self._grads), self._adapt_grad_clip_thresh),
              lambda: self._adapt_grad_clip_target_val,
              lambda: tf.to_float(tf.constant(LARGE_FLOAT_VAL)))
          self._grads, self._grads_norm = tf.clip_by_global_norm(
              self._grads, thresh)

        apply_grad_op = self._optimizer.apply_gradients(
            zip(self._grads, self._tvars), global_step, name)

    with tf.control_dependencies([apply_grad_op]):
      self._increment_global_step_op = tf.assign(self._global_step,
                                                 self._global_step + 1)

      self._adapt_grad_clip_thresh_op = \
        tf.assign(self._adapt_grad_clip_thresh, tf.sqrt(self._h_max) )
      self._adapt_grad_clip_target_val_op = \
        tf.assign(self._adapt_grad_clip_target_val, tf.sqrt(self._h_max) )
      # self._adapt_grad_clip_target_val_op = \
      #   tf.assign(self._adapt_grad_clip_target_val, tf.sqrt(tf.sqrt(self._h_max * self._h_min)))

    return tf.group(before_apply_op, update_hyper_op, apply_grad_op,
                    self._adapt_grad_clip_thresh_op,
                    self._adapt_grad_clip_target_val_op,
                    self._increment_global_step_op)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    return self._optimizer.compute_gradients(
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=False,
               name=None,
               grad_loss=None):
    """Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before
    applying them, call `tf.gradients()` and `self.apply_gradients()`
    explicitly instead of using this function.

    Adapted from Tensorflow Optimizer base class member function.
    """
    grads_and_vars = self._optimizer.compute_gradients(
        loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for "
          "ops that do not support gradients, between variables "
          "%s and loss %s." % ([str(v) for _, v in grads_and_vars], loss))

    return self.apply_gradients(grads_and_vars, global_step, name)

  def get_slot(self, var, name):
    """
    Return a slot named `name` created for `var` by
    the underlying MomentumOptimizer.

    Args:
      var: A variable passed to `minimize()` or `apply_gradients()`.
      name: A string.

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._optimizer.get_slot(var, name)

  def get_slot_names(self):
    """
    Return a list of the names of the slots created by the
    underlying MomentumOptimizer.

    Returns:
      A list of strings.
    """
    return self._optimizer.get_slot_names()
