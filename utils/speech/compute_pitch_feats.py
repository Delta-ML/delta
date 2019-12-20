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
"""Create Pitch feature files."""

import delta.compat as tf
import argparse
from distutils.util import strtobool
import kaldiio
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from delta.data.frontend.pitch import Pitch
from espnet.utils.cli_writers import KaldiWriter


def get_parser():
  parser = argparse.ArgumentParser(
      description='Compute pitch features from wav.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--sample_rate', type=int, default=16000, help='Sampling frequency')
  parser.add_argument(
      '--window_length', type=float, default=0.025, help='Length of a frame')
  parser.add_argument(
      '--frame_length', type=float, default=0.010, help='Hop size of window')
  parser.add_argument(
      '--snip_edges',
      type=bool,
      default=True,
      help='The last frame (shorter than window_length) will not be cutoff.')
  parser.add_argument(
      '--preemph_coeff',
      type=float,
      default=0.0,
      help='Coefficient for use in frame-signal preemphasis.')
  parser.add_argument(
      '--min_f0', type=float, default=50, help='F0 to search for (Hz).')
  parser.add_argument(
      '--max_f0', type=float, default=400, help='F0 to search for (Hz).')
  parser.add_argument(
      '--soft_min_f0',
      type=float,
      default=10.0,
      help='Minimum f0, applied in soft way, must not exceed min-f0.')
  parser.add_argument(
      '--penalty_factor',
      type=float,
      default=0.1,
      help='cost factor for FO change.')
  parser.add_argument(
      '--lowpass_cutoff',
      type=float,
      default=1000,
      help='cutoff frequency for LowPass filter (Hz).')
  parser.add_argument(
      '--resample_freq',
      type=float,
      default=4000.0,
      help='Frequency that we down-sample the signal to.  Must be more than twice lowpass-cutoff.'
  )
  parser.add_argument(
      '--delta_pitch',
      type=float,
      default=0.005,
      help='Smallest relative change in pitch that our algorithm measures.')
  parser.add_argument(
      '--nccf_ballast',
      type=float,
      default=7000.0,
      help='Increasing this factor reduces NCCF for quiet frames.')
  parser.add_argument(
      '--lowpass_filter_width',
      type=int,
      default=1,
      help='Integer that determines filter width of lowpass filter, more gives sharper filter.'
  )
  parser.add_argument(
      '--upsample_filter_width',
      type=int,
      default=5,
      help='Integer that determines filter width when upsampling NCCF.')
  parser.add_argument(
      '--max_frames_latency',
      type=int,
      default=0,
      help='Maximum number of frames of latency that we allow pitch tracking to introduce into the feature processing.'
  )
  parser.add_argument(
      '--frames_per_chunk',
      type=int,
      default=0,
      help='Only relevant for offline pitch extraction.')
  parser.add_argument(
      '--recompute_frame',
      type=int,
      default=500,
      help='Only relevant for online pitch extraction, or for compatibility with online pitch extraction.'
  )
  parser.add_argument(
      '--simulate_first_pass_online',
      type=bool,
      default=False,
      help='If true, compute-kaldi-pitch-feats will output features that correspond to what an '
      'online decoder would see in the first pass of decoding.')
  parser.add_argument(
      '--nccf_ballast_online',
      type=bool,
      default=False,
      help='This is useful mainly for debug; it affects how the NCCF ballast is computed.'
  )
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


def compute_pitch():
  parser = get_parser()
  args = parser.parse_args()

  config = {}
  config['sample_rate'] = int(args.sample_rate)
  config['window_length'] = args.window_length
  config['frame_length'] = args.frame_length
  config['snip_edges'] = args.snip_edges
  config['preemph_coeff'] = args.preemph_coeff
  config['min_f0'] = args.min_f0
  config['max_f0'] = args.max_f0
  config['soft_min_f0'] = args.soft_min_f0
  config['penalty_factor'] = args.penalty_factor
  config['lowpass_cutoff'] = args.lowpass_cutoff
  config['resample_freq'] = args.resample_freq
  config['delta_pitch'] = args.delta_pitch
  config['nccf_ballast'] = args.nccf_ballast
  config['lowpass_filter_width'] = args.lowpass_filter_width
  config['upsample_filter_width'] = args.upsample_filter_width
  config['max_frames_latency'] = args.max_frames_latency
  config['frames_per_chunk'] = args.frames_per_chunk
  config['simulate_first_pass_online'] = args.simulate_first_pass_online
  config['recompute_frame'] = args.recompute_frame
  config['nccf_ballast_online'] = args.nccf_ballast_online

  pitch = Pitch.params(config).instantiate()

  with kaldiio.ReadHelper(args.rspecifier,
                          segments=args.segments) as reader, \
        KaldiWriter(args.wspecifier, write_num_frames=args.write_num_frames,
                    compress=args.compress, compression_method=args.compression_method) as writer:
    for utt_id, (sample_rate, array) in reader:
      if sample_rate != args.sample_rate:
        args.sample_rate = sample_rate
      array = array.astype(np.float32)
      audio_data = tf.constant(array, dtype=tf.float32)
      pitch_test = tf.squeeze(pitch(audio_data, args.sample_rate))
      sess = tf.Session()
      pitch_feats = pitch_test.eval(session=sess)
      writer[utt_id] = pitch_feats


if __name__ == "__main__":
  compute_pitch()
