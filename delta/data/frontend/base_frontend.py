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
''' base interface of Frontend '''

import abc
import delta.compat as tf

from delta.utils.hparam import HParams


class ABCFrontend(metaclass=abc.ABCMeta):
  ''' abstract of Frontend '''

  def __init__(self, config):
    raise NotImplementedError()

  @abc.abstractmethod
  def call(self, *args, **kwargs):
    ''' implementation func '''
    raise NotImplementedError()


class BaseFrontend(ABCFrontend):
  ''' wrapper of abstrcat Frontend'''

  def __init__(self, config: dict):
    self._config = config

  @property
  def config(self):
    ''' config property '''
    return self._config

  @classmethod
  def params(cls, config=None):
    ''' set params '''
    raise NotImplementedError()

  def __call__(self, *args, **kwargs):
    ''' call '''
    return self.call(*args, **kwargs)
