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
from delta.data.frontend.analyfiltbank import Analyfiltbank
from espnet.utils.cli_writers import KaldiWriter


def get_parser():
  parser = argparse.ArgumentParser(
      description='Compute power specturm or phase specturm features from wav.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--sample_rate', type=int, default=16000, help='Sampling frequency')
  parser.add_argument(
      '--window_length', type=float, default=0.030, help='Length of a frame')
  parser.add_argument(
      '--frame_length', type=float, default=0.010, help='Hop size of window')
  parser.add_argument(
      '--output_type',
      type=int,
      default=1,
      help='1 for power spectrum, 2 for phase spectrum.')
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


def compute_stft():
  parser = get_parser()
  args = parser.parse_args()

  config = {}
  config['sample_rate'] = int(args.sample_rate)
  config['window_length'] = args.window_length
  config['frame_length'] = args.frame_length

  stft = Analyfiltbank.params(config).instantiate()

  with kaldiio.ReadHelper(args.rspecifier,
                          segments=args.segments) as reader, \
        KaldiWriter(args.wspecifier, write_num_frames=args.write_num_frames,
                    compress=args.compress, compression_method=args.compression_method) as writer:
    for utt_id, (sample_rate, array) in reader:
      if sample_rate != args.sample_rate:
        args.sample_rate = sample_rate
      array = array.astype(np.float32)
      audio_data = tf.constant(array, dtype=tf.float32)
      power_spectrum, phase_spectrum = stft(audio_data, args.sample_rate)
      sess = tf.Session()
      if args.output_type == 1:
        out_feats = power_spectrum.eval(session=sess)
      else:
        out_feats = phase_spectrum.eval(session=sess)
      writer[utt_id] = out_feats


if __name__ == "__main__":
  compute_stft()
