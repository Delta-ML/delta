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
''' postprocess abstract class '''
import abc


#pylint: disable=too-few-public-methods
class PostProcABC(metaclass=abc.ABCMeta):
  ''' postprocess abstract class'''

  def __init__(self, config):
    pass

  @abc.abstractmethod
  def call(self):
    ''' implementation func '''
    raise NotImplementedError()


#pylint: disable=abstract-method
class PostProc(PostProcABC):
  ''' base class of postprocess class'''

  def __init__(self, config):
    super().__init__(config)
    self.config = config

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)
