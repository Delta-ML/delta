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

import io
import kaldiio
import numpy as np
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend

# This version is efficient, but without hparams.
# class CMVN(object):
#   def __init__(self, stats, norm_means=True, norm_vars=False,
#                utt2spk=None, spk2utt=None, reverse=False, std_floor=1.0e-20):
#     self.stats_file = stats
#     self.norm_means = norm_means
#     self.norm_vars = norm_vars
#     self.reverse = reverse
#
#     if isinstance(stats, dict):
#       stats_dict = dict(stats)
#     else:
#       self.accept_uttid = True
#       stats_dict = dict(kaldiio.load_ark(stats))
#
#     if utt2spk is not None:
#       self.utt2spk = {}
#       with io.open(utt2spk, 'r', encoding='utf-8') as f:
#         for line in f:
#           utt, spk = line.rstrip().split(None, 1)
#           self.utt2spk[utt] = spk
#     elif spk2utt is not None:
#       self.utt2spk = {}
#       with io.open(spk2utt, 'r', encoding='utf-8') as f:
#         for line in f:
#           spk, utts = line.rstrip().split(None, 1)
#           for utt in utts.split():
#             self.utt2spk[utt] = spk
#     else:
#       self.utt2spk = None
#
#     self.bias = {}
#     self.scale = {}
#     for spk, stats in stats_dict.items():
#       assert len(stats) == 2, stats.shape
#
#       count = stats[0, -1]
#
#       if not (np.isscalar(count) or isinstance(count, (int, float))):
#         count = count.flatten()[0]
#
#       mean = stats[0, :-1] / count
#       var = stats[1, :-1] / count - mean * mean
#       std = np.maximum(np.sqrt(var), std_floor)
#       self.bias[spk] = -mean
#       self.sacle[spk] = 1 / std
#
#   def __repr__(self):
#     return ('{name}(stats_file={stats_file}, '
#             'norm_means={norm_means}, norm_vars={norm_vars}, '
#             'reverse={reverse})'
#             .format(name=self.__class__.__name__,
#                     stats_file=self.stats_file,
#                     norm_means=self.norm_means,
#                     norm_vars=self.norm_vars,
#                     reverse=self.reverse))
#
#   def __call__(self, x, uttid=None):
#     if self.utt2spk is not  None:
#       spk = self.utt2spk[uttid]
#     else:
#       spk = uttid
#
#     if not self.reverse:
#       if self.norm_means:
#         x = np.add(x, self.bias[spk])
#       if self.norm_vars:
#         x = np.multiply(x, self.scale[spk])
#     else:
#       if self.norm_means:
#         x = np.subtract(x, self.bias[spk])
#       if self.norm_vars:
#         x = np.divide(x, self.scale[spk])
#
#     return x

class CMVN(BaseFrontend):

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):

    norm_means = True
    norm_vars = False
    utt2spk = None
    spk2utt = None
    reverse = False
    std_floor = 1.0e-20

    hparams = HParams(cls=cls)
    hparams.add_hparam('norm_means', norm_means)
    hparams.add_hparam('norm_vars', norm_vars)
    hparams.add_hparam('utt2spk', utt2spk)
    hparams.add_hparam('spk2utt', spk2utt)
    hparams.add_hparam('reverse', reverse)
    hparams.add_hparam('std_floor', std_floor)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, stats, x, uttid=None):

    p = self.config

    # The way is not efficient.
    if isinstance(stats, dict):
      stats_dict = dict(stats)
    else:
      stats_dict = dict(kaldiio.load_ark(stats))

    if p.utt2spk is not None:
      self.utt2spk = {}
      with io.open(p.utt2spk, 'r', encoding='utf-8') as f:
        for line in f:
          utt, spk = line.rstrip().split(None, 1)
          self.utt2spk[utt] = spk
    elif p.spk2utt is not None:
      self.utt2spk = {}
      with io.open(p.spk2utt, 'r', encoding='utf-8') as f:
        for line in f:
          spk, utts = line.rstrip().split(None, 1)
          for utt in utts.split():
            self.utt2spk[utt] = spk
    else:
      self.utt2spk = None

    self.bias = {}
    self.scale = {}
    for spk, stats in stats_dict.items():
      assert len(stats) == 2, stats.shape

      count = stats[0, -1]

      if not (np.isscalar(count) or isinstance(count, (int, float))):
        count = count.flatten()[0]

      mean = stats[0, :-1] / count
      var = stats[1, :-1] / count - mean * mean
      std = np.maximum(np.sqrt(var), p.std_floor)
      self.bias[spk] = -mean
      self.sacle[spk] = 1 / std

    if self.utt2spk is not  None:
      spk = self.utt2spk[uttid]
    else:
      spk = uttid

    if not p.reverse:
      if p.norm_means:
        x = np.add(x, self.bias[spk])
      if p.norm_vars:
        x = np.multiply(x, self.scale[spk])
    else:
      if p.norm_means:
        x = np.subtract(x, self.bias[spk])
      if p.norm_vars:
        x = np.divide(x, self.scale[spk])

    return x
