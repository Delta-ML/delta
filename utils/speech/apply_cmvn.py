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
from distutils.util import strtobool
import kaldiio
import numpy as np
from espnet.utils.cli_writers import KaldiWriter
from espnet.utils.cli_readers import KaldiReader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from delta.data.frontend.cmvn import CMVN

def get_parser():
  parser = argparse.ArgumentParser(
    description='Apply mean-variance normalization to files',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
    '--norm_means', type=strtobool, default=True,
    help='Do mean normalization or not.')
  parser.add_argument(
    '--norm_vars', type=strtobool, default=False,
    help='Do variance normalization or not.')
  parser.add_argument(
    '--reverse', type=strtobool, default=False,
    help='Do reverse mode or not')
  parser.add_argument(
    '--spk2utt', type=str,
    help='A text file of speaker to utterance-list map. '
    '(Don\'t give rspecifier format, such as "ark:spk2utt")')
  parser.add_argument(
    '--utt2spk', type=str,
    help='A text file of utterance to speaker map. '
    '(Don\'t give rspecifier format, such as "ark:utt2spk")')
  parser.add_argument(
    '--write_num_frames', type=str,
    help='Specify wspecifer for utt2num_frames')
  parser.add_argument(
    '--compress', type=strtobool, default=False,
    help='Save data in compressed format')
  parser.add_argument(
    '--compression_method', type=int, default=2,
    help='Specify the method of compression')
  parser.add_argument(
    '--verbose', '-V', default=0, type=int, help='Verbose option')
  parser.add_argument(
    'stats_rspecifier',
    help='Input stats. e.g. ark:stats.ark')
  parser.add_argument(
    'rspecifier', type=str,
    help='Read specifier id. e.g. scp:some.scp')
  parser.add_argument(
    'wspecifier', type=str,
    help='Write specifier id. e.g. ark:some.ark')

  return parser

def apply_cmvn():
  args = get_parser().parse_args()

  config = {}
  config['norm_means'] = args.norm_means
  config['norm_vars'] = args.norm_vars
  config['utt2spk'] = args.utt2spk
  config['spk2utt'] = args.spk2utt
  config['reverse'] = args.reverse
  config['std_floor'] = args.std_floor
  cmvn = CMVN.params(config).instantiate()

  stats_dict = dict(KaldiReader(args.stats_rspecifier, 'mat'))
  with KaldiWriter(args.wspecifier, write_num_frames=args.write_num_frames,
                compress=args.compress, compression_method=args.compression_method) as writer:
    for utt, mat in KaldiReader(args.rspecifier):
      mat = cmvn(stats_dict, mat, utt)
      writer[utt] = mat

if __name__ == '__main__':
  apply_cmvn()
