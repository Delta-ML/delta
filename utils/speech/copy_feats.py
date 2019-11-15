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
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.cli_readers import KaldiReader
import kaldiio


def get_parser():
  parser = argparse.ArgumentParser(
      description='copy feature with preprocessing',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--verbose', '-V', default=0, type=int, help='Verbose option')
  parser.add_argument(
      '--write_num_frames',
      type=str,
      help='Specify wspecifer for utt2num_frames')
  parser.add_argument(
      '--compress',
      type=strtobool,
      default=False,
      help='Save in compressed format')
  parser.add_argument(
      '--compression_method',
      type=int,
      default=2,
      help='Specify the method(if mat) or gzip-level(if hdf5)')
  parser.add_argument(
      'rspecifier',
      type=str,
      help='Read specifier for feats. e.g. ark:some.ark')
  parser.add_argument(
      'wspecifier', type=str, help='Write specifier. e.g. ark:some.ark')
  return parser


def main():
  parser = get_parser()
  args = parser.parse_args()

  d = kaldiio.load_ark(args.rspecifier)

  with file_writer_helper(
      args.wspecifier,
      filetype='mat',
      write_num_frames=args.write_num_frames,
      compress=args.compress,
      compression_method=args.compression_method) as writer:
    for utt, mat in d:
      writer[utt] = mat


if __name__ == "__main__":
  main()
