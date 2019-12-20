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
"""The model computes CMVN of features."""

import io
import kaldiio
import numpy as np
from delta.utils.hparam import HParams
from delta.data.frontend.base_frontend import BaseFrontend


class CMVN(BaseFrontend):
  """
  Compute and apply CMVN to features.
  """

  def __init__(self, config: dict):
    super().__init__(config)

  @classmethod
  def params(cls, config=None):
    """
    Set params.
    :param config: contains seven optional parameters:
            --norm_means   : Flag of norm_means. (bool, default=True)
            --norm_vars    : Flag of norm_vars. (bool, default=False)
            --utt2spk      : Use for speaker CMVN. (string, default=None)
            --spk2utt      : Rspecifier for speaker to utterance-list map.
                            (string, default=None)
            --reverse      : Flag of reverse. (bool, default=False)
            --std_floor    : Floor to std. (float, default=1.0e-20)
            --filetype     : Type of input file. (string, default='mat')
    :return:
    """
    norm_means = True
    norm_vars = False
    utt2spk = None
    spk2utt = None
    reverse = False
    std_floor = 1.0e-20
    filetype = 'mat'

    hparams = HParams(cls=cls)
    hparams.add_hparam('norm_means', norm_means)
    hparams.add_hparam('norm_vars', norm_vars)
    hparams.add_hparam('utt2spk', utt2spk)
    hparams.add_hparam('spk2utt', spk2utt)
    hparams.add_hparam('reverse', reverse)
    hparams.add_hparam('std_floor', std_floor)
    hparams.add_hparam('filetype', filetype)

    if config is not None:
      hparams.override_from_dict(config)

    return hparams

  def call(self, stats):
    """
    Do CMVN.
    :param stats: Statistics of features.
    :return: Mean and std of features.
    """
    p = self.config

    if isinstance(stats, dict):
      stats_dict = dict(stats)
    else:
      if p.filetype == 'mat':
        stats_dict = {None: kaldiio.load_mat(stats)}
      elif p.filetype == 'ark':
        stats_dict = dict(kaldiio.load_ark(stats))
      else:
        raise ValueError('Not supporting filetype={}'.format(p.filetype))

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
      self.scale[spk] = 1 / std

  def apply_cmvn(self, x, uttid):

    p = self.config

    if self.utt2spk is not None:
      spk = self.utt2spk[uttid]
    else:
      # using global cmvn
      spk = None

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
