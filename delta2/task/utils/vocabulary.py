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
"""Going to be deprecated"""
import copy
import collections

# pylint: disable=too-many-instance-attributes


class Vocabulary:
  ''' vocabulary '''

  def __init__(self, use_default_dict):
    self._padding_token = "<pad>"
    self._unknown_token = "<unk>"
    self._start_of_sentence = "<sos>"
    self._end_of_sentence = "<eos>"
    self._s_token = "<s>"
    self._slash_s_token = "</s>"
    self._default_dict = {
        self._padding_token: 0,
        self._s_token: 1,
        self._slash_s_token: 2,
        self._unknown_token: 3,
        self._start_of_sentence: 4,
        self._end_of_sentence: 5
    }
    self.use_default_dict = use_default_dict
    if self.use_default_dict:
      self._mapping = copy.deepcopy(self._default_dict)
    else:
      self._mapping = {}
    self._freq = collections.defaultdict(int)

  def __getitem__(self, key):
    return self._mapping[key]

  def add(self, word):
    ''' update vocab statis'''
    if word not in self._mapping:
      self._mapping[word] = len(self._mapping)
    self._freq[word] += 1

  def trim(self, min_frequency):
    ''' trim word freq less than min_frequency'''
    # sort by frequency
    self._freq = sorted(self._freq.items(), key=lambda x: x[1], reverse=True)

    if self.use_default_dict:
      self._mapping = copy.deepcopy(self._default_dict)
      idx = len(self._default_dict)
    else:
      self._mapping = {}
      idx = 0

    for word, count in self._freq:
      if count < min_frequency:
        break
      if word in self._mapping:
        continue
      self._mapping[word] = idx
      idx += 1
    self._freq = dict(self._freq[:idx - 1])

  @property
  def freq(self):
    '''candy _freq'''
    return self._freq

  @property
  def mapping(self):
    ''' candy _mapping'''
    return self._mapping
