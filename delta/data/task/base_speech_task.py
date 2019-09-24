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
''' Base Speech Task'''
from delta import utils
from delta.data import utils as data_utils
from delta.data.task.base_task import WavSpeechTask

#pylint: disable=abstract-method


class SpeechTask(WavSpeechTask):
  ''' base class for speech task'''

  def __init__(self, config, mode):
    super().__init__(config)
    assert mode in (utils.TRAIN, utils.EVAL, utils.INFER)
    self._mode = mode

  @property
  def mode(self):
    return self._mode

  #pylint: disable=arguments-differ
  def input_fn(self, mode, batch_size, num_epoch=None):
    ''' estimator input_fn'''
    return data_utils.input_fn(self.dataset, mode, batch_size, num_epoch)
