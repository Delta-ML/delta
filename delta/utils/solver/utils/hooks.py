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
"""Hooks"""

import delta.compat as tf
from bisect import bisect_right
from absl import logging


class DatasetInitializerHook(tf.estimator.SessionRunHook):
  ''' iterator dataset initailizer '''

  def __init__(self, iterator, init_feed_dict):
    self._iterator = iterator
    self._init_feed_dict = init_feed_dict

  def begin(self):
    self._initializer = self._iterator.initializer

  def after_create_session(self, session, coord):
    del coord
    session.run(self._initializer, self._init_feed_dict)


class EpochHook(tf.estimator.SessionRunHook):

  def __init__(self, examples_per_epoch, global_batch_size):
    self._num_examples_per_epoch = examples_per_epoch
    self._global_batch_size = global_batch_size
    self._epoch = 0

  @property
  def epoch(self):
    return self._epoch

  def begin(self):
    self._global_step_tensor = tf.train.get_or_create_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use StopAtStepHook.")
    self._epoch_tensor = (self._global_step_tensor * tf.constant(
        self._num_examples_per_epoch)) / tf.constant(self._global_batch_size)

  def after_create_session(self, session, coord):
    pass

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results + 1

    # Check latest global step to ensure that the targeted last step is
    # reached. global_step read tensor is the value of global step
    # before running the operation. We're not sure whether current session.run
    # incremented the global_step or not. Here we're checking it.

    step = run_context.session.run(self._global_step_tensor)
    assert step == global_step
    self._epoch = int(
        (self._global_batch_size * step) / self._num_examples_per_epoch)
    logging.info(f"{self.__class__.__name__}: Epoch {self.epoch}")

  def end(self, session):
    pass


class MultiStepLRHook(tf.estimator.SessionRunHook):
  ''' Set the learning rate of each parameter group to the initial lr decayed 
      by gamma once the number of epoch reaches one of the milestones. 
      When last_epoch=-1, sets initial lr as lr.
  params:
    lr (flaot) : init learning rate
    milestones (list) : List of epoch indices. Must be increasing.
    gamma (float) : Multiplicative factor of learning rate decay. Default: 0.1.
    last_epoch (int) : The index of last epoch. Default: -1.
  '''

  def __init__(self, lr, milestones, gamma=0.1, last_epoch=-1):
    if not list(milestones) == sorted(milestones):
      raise ValueError(
          'Milestones should be a list of'
          ' increasing integers. Got {}', milestones)
    self._milestones = milestones
    self._lrn_rate = lr
    self._gamma = gamma
    self._last_epoch = last_epoch

  def begin(self):
    self._global_step_tensor = tf.train.get_or_create_global_step()
    self._lrn_rate_tensor = tf.get_default_graph().get_tensor_by_name(
        'learning_rate:0')

  def after_create_session(self, session, coord):
    pass

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(
        self._global_step_tensor,  # Asks for global step value.
        feed_dict={self._lrn_rate_tensor: self._lrn_rate})  # Sets learning rate

  def after_run(self, run_context, run_values):
    train_step = run_values.results
    self._lrn_rate = self.get_lr()

  def get_lr(self):
    return self._lrn_rate * self._gamma**bisect_right(self._milestones,
                                                      self.last_epoch)

  def end(self, session):
    pass
