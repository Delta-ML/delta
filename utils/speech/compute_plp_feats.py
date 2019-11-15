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

import delta.compat as tf
import argparse
from distutils.util import strtobool
import kaldiio
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from delta.data.frontend.plp import Plp
from espnet.utils.cli_writers import KaldiWriter


def get_parser():
  parser = argparse.ArgumentParser(
      description='Compute plp features from wav.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--sample_rate', type=int, default=16000, help='Sampling frequency')
  parser.add_argument('--plp_order', type=int, default=12, help='Order of plp')
  parser.add_argument(
      '--window_length', type=float, default=0.025, help='Length of a frame')
  parser.add_argument(
      '--frame_length', type=float, default=0.010, help='Hop size of window')
  parser.add_argument(
      '--write_num_frames',
      type=str,
      help='Specify wspecifer for utt2num_frames')
  parser.add_argument(
      '--compress',
      type=strtobool,
      default=False,
      help='Save data in compressed format')
  parser.add_argument(
      '--compression_method',
      type=int,
      default=2,
      help='Specify the method of compression')
  parser.add_argument(
      '--verbose', '-V', default=0, type=int, help='Verbose option')
  parser.add_argument(
      '--segments',
      type=str,
      help='segments-file format: each line is either'
      '<segment-id> <recording-id> <start-time> <end-time>'
      'e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5')
  parser.add_argument('rspecifier', type=str, help='WAV scp file')
  parser.add_argument('wspecifier', type=str, help='Writer specifier')
  return parser


def compute_plp():
  parser = get_parser()
  args = parser.parse_args()

  config = {}
  config['sample_rate'] = int(args.sample_rate)
  config['plp_order'] = int(args.plp_order)
  config['window_length'] = args.window_length
  config['frame_length'] = args.frame_length

  plp = Plp.params(config).instantiate()

  with kaldiio.ReadHelper(args.rspecifier,
                          segments=args.segments) as reader, \
        KaldiWriter(args.wspecifier, write_num_frames=args.write_num_frames,
                    compress=args.compress, compression_method=args.compression_method) as writer:
    for utt_id, (sample_rate, array) in reader:
      if sample_rate != args.sample_rate:
        args.sample_rate = sample_rate
      array = array.astype(np.float32)
      audio_data = tf.constant(array, dtype=tf.float32)
      plp_test = plp(audio_data, args.sample_rate)
      sess = tf.Session()
      plp_feats = plp_test.eval(session=sess)
      writer[utt_id] = plp_feats


if __name__ == "__main__":
  compute_plp()
