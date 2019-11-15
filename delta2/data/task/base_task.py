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
''' Task abstract class for data process'''
import abc


class Task(metaclass=abc.ABCMeta):
  ''' abstract class'''

  def __init__(self, config):
    self._config = config

  @property
  def config(self):
    ''' config property'''
    return self._config

  @abc.abstractmethod
  def generate_data(self):
    ''' generate one example'''
    raise NotImplementedError()

  @abc.abstractmethod
  def feature_spec(self):
    ''' dataset meta data'''
    raise NotImplementedError()

  @abc.abstractmethod
  def preprocess_batch(self, batch):
    ''' pre-proecss of data'''
    raise NotImplementedError()

  @abc.abstractmethod
  def dataset(self):
    ''' generate batch examples with epoch
    return tf.data.Dataset
    '''
    return NotImplementedError()

  @abc.abstractmethod
  def input_fn(self):
    ''' return `def _input_fn()` function'''
    return NotImplementedError()


class WavSpeechTask(Task):
  ''' Speech task which need generate feat and cmvn'''

  @abc.abstractmethod
  def generate_feat(self, paths):
    '''
        generate features,
           paths: list or tuple
        '''
    return NotImplementedError()

  @abc.abstractmethod
  def generate_cmvn(self, paths):
    '''
        generate mean and vars of features,
            paths: list or tuple
        '''
    return NotImplementedError()
