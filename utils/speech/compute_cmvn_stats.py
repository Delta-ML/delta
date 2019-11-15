#!/usr/bin/env python3

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

import argparse
import logging
import kaldiio
import numpy as np
from espnet.utils.cli_writers import KaldiWriter
from espnet.utils.cli_readers import KaldiReader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_parser():
  parser = argparse.ArgumentParser(
      description='Compute cepstral mean and variance normalization statistics'
      'per-utterance by default, or per-speaker if spk2utt option provided,'
      'if wxfilename: global',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--spk2utt',
      type=str,
      default=None,
      help='A text file of speaker to utterance-list map. '
      '(Don\'t give rspecifier format, such as "ark:spk2utt")')
  parser.add_argument(
      '--verbose', '-V', default=0, type=int, help='Verbose option')
  parser.add_argument(
      'rspecifier', type=str, help='Read specifier id. e.g. scp:some.scp')
  parser.add_argument(
      'wspecifier_or_wxfilename',
      type=str,
      help='Write specifier id. e.g. ark:some.ark')
  return parser


def compute_cmvn_stats():
  """
  e.g. compute_cmvn_stats.py scp:data/train/feats.scp data/train/cmvn.ark # compute global cmvn
  """
  args = get_parser().parse_args()

  is_wspecifier = ':' in args.wspecifier_or_wxfilename

  if is_wspecifier:
    if args.spk2utt is not None:
      utt2spk_dict = {}
      with open(args.spk2utt) as f:
        for line in f:
          spk, utts = line.rstrip().split(None, 1)
          for utt in utts.split():
            utt2spk_dict[utt] = spk

      def utt2spk(x):
        return utt2spk_dict[x]
    else:
      logging.info('Performing as utterance CMVN mode')

      def utt2spk(x):
        return x

  else:
    logging.info('Performing as gloabl CMVN model')
    if args.spk2utt is not None:
      logging.warning('spk2utt is not used for global CMVN mode')

    def utt2spk(x):
      return None

  # Calculate stats for each speaker
  counts = {}
  sum_feats = {}
  square_sum_feats = {}

  idx = 0
  for idx, (utt, matrix) in enumerate(KaldiReader(args.rspecifier), 1):
    spk = utt2spk(utt)

    if spk not in counts:
      counts[spk] = 0
      feat_shape = matrix.shape[1:]
      sum_feats[spk] = np.zeros(feat_shape, dtype=np.float)
      square_sum_feats[spk] = np.zeros(feat_shape, dtype=np.float)

    counts[spk] += matrix.shape[0]
    sum_feats[spk] += matrix.sum(axis=0)
    square_sum_feats[spk] += (matrix**2).sum(axis=0)

  assert idx > 0, idx

  cmvn_stats = {}
  for spk in counts:
    feat_shape = sum_feats[spk].shape
    cmvn_shape = (2, feat_shape[0] + 1) + feat_shape[1:]
    _cmvn_stats = np.empty(cmvn_shape, dtype=np.float64)
    _cmvn_stats[0, :-1] = sum_feats[spk]
    _cmvn_stats[1, :-1] = square_sum_feats[spk]

    _cmvn_stats[0, -1] = counts[spk]
    _cmvn_stats[1, -1] = 0.

    cmvn_stats[spk] = _cmvn_stats

  if is_wspecifier:
    with KaldiWriter(args.wspecifier_or_wxfilename) as writer:
      for spk, mat in cmvn_stats.items():
        writer[spk] = mat
  else:
    matrix = cmvn_stats[None]
    kaldiio.save_mat(args.wspecifier_or_wxfilename, matrix)


if __name__ == "__main__":
  compute_cmvn_stats()
