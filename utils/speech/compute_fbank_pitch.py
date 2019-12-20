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
"""Create fbank_picth feature files."""

import delta.compat as tf
import argparse
from distutils.util import strtobool
import kaldiio
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from delta.data.frontend.fbank_pitch import FbankPitch
from espnet.utils.cli_writers import KaldiWriter


def get_parser():
  parser = argparse.ArgumentParser(
      description='Compute fbank && pitch feature from wav.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--sample_rate', type=float, default=16000, help='Sampling frequency')
  parser.add_argument(
      '--upper_frequency_limit',
      type=float,
      default=4000,
      help='Maxinum frequency')
  parser.add_argument(
      '--lower_frequency_limit',
      type=float,
      default=20,
      help='Minimum frequency')
  parser.add_argument(
      '--filterbank_channel_count',
      type=float,
      default=40,
      help='Order of fbank')
  parser.add_argument(
      '--dither',
      type=float,
      default=0.0,
      help='Dithering constant (0.0 means no dither).')
  parser.add_argument(
      '--window_length', type=float, default=0.025, help='Length of a frame')
  parser.add_argument(
      '--frame_length', type=float, default=0.010, help='Hop size of window')
  parser.add_argument(
      '--output_type',
      type=int,
      default=1,
      help='1 for power spectrum, 2 for log-power spectrum.')
  parser.add_argument(
      '--window_type',
      type=str,
      default='povey',
      help='Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria").')
  parser.add_argument(
      '--snip_edges',
      type=bool,
      default=True,
      help='The last frame (shorter than window_length) will not be cutoff.')
  parser.add_argument(
      '--raw_energy',
      type=int,
      default=1,
      help='Compute frame energy before preemphasis and windowing.')
  parser.add_argument(
      '--preeph_coeff',
      type=float,
      default=0.97,
      help='Coefficient for use in frame-signal preemphasis.')
  parser.add_argument(
      '--remove_dc_offset',
      type=bool,
      default=True,
      help=' Subtract mean from waveform on each frame')
  parser.add_argument(
      '--is_fbank',
      type=bool,
      default=True,
      help='Compute power spetrum without frame energy')
  parser.add_argument(
      '--thres_autoc', type=float, default=0.3, help='Threshold of autoc')
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


def compute_fbank_pitch():
  parser = get_parser()
  args = parser.parse_args()

  config = {}
  config['sample_rate'] = int(args.sample_rate)
  config['upper_frequency_limit'] = float(args.upper_frequency_limit)
  config['lower_frequency_limit'] = float(args.lower_frequency_limit)
  config['filterbank_channel_count'] = float(args.filterbank_channel_count)
  config['window_length'] = args.window_length
  config['frame_length'] = args.frame_length
  config['output_type'] = int(args.output_type)
  config['window_type'] = args.window_type
  config['snip_edges'] = args.snip_edges
  config['preeph_coeff'] = args.preeph_coeff
  config['remove_dc_offset'] = args.remove_dc_offset
  config['is_fbank'] = args.is_fbank
  config['thres_autoc'] = args.thres_autoc
  config['dither'] = args.dither

  fbank_pitch = FbankPitch.params(config).instantiate()

  with kaldiio.ReadHelper(args.rspecifier,
                          segments=args.segments) as reader, \
        KaldiWriter(args.wspecifier, write_num_frames=args.write_num_frames,
                    compress=args.compress, compression_method=args.compression_method) as writer:
    for utt_id, (sample_rate, array) in reader:
      if sample_rate != args.sample_rate:
        args.sample_rate = sample_rate
      array = array.astype(np.float32)
      audio_data = tf.constant(array, dtype=tf.float32)
      fbank_pitch_test = fbank_pitch(audio_data, args.sample_rate)
      sess = tf.Session()
      fbank_pitch_feats = fbank_pitch_test.eval(session=sess)
      writer[utt_id] = fbank_pitch_feats


if __name__ == "__main__":
  compute_fbank_pitch()
