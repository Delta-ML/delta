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
''' base interface of loss '''
import abc
import delta.compat as tf

from delta.utils import summary


class ABCLoss(metaclass=abc.ABCMeta):  #pylint: disable=too-few-public-methods
  ''' abstract of Loss '''
  #pylint: disable=too-many-arguments
  @abc.abstractmethod
  def call(self,
           logits=None,
           input_length=None,
           labels=None,
           label_length=None,
           **kwargs):
    '''
    param: logits, (B, T, D)
    param: labels, (B, T)
    param: label_length, (B), converts labels form dense to sparse
    param: input_length, (B), input length of encoder
    returns: loss, scalar
    '''
    raise NotImplementedError()


class Loss(ABCLoss):  #pylint: disable=abstract-method
  ''' wappwer of abstrcat Loss '''

  def __init__(self, config):
    self._config = config

  @property
  def config(self):
    ''' config property '''
    return self._config

  def __call__(self, **kwargs):
    name = kwargs.get('name')
    kwargs.pop('name')
    with tf.variable_scope(name):
      loss = self.call(**kwargs)
    summary.scalar(name, loss)
    return loss
