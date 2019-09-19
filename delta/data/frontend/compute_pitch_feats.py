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
from distutils.util import strtobool

import kaldiio
import numpy as np

from delta.data.frontend.pitch import Pitch
from delta.data.frontend.kaldi_delta_io import init_kaldi_writer

def get_parser():
  parser = argparse.ArgumentParser(
    description='compute pitch features from wav file',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--window_length', type=float, default=0.025,
                      help='The window length of FFT')
  parser.add_argument('--frame_length', type=float, default=0.010,
                      help='The hopsize of FFT')
  parser.add_argument('--thres_autoc', type=float, default=0.3,
                      help='The thres_autoc of pitch')
  parser.add_argument('--write_num_frames', type=str,
                      help='Specify wspecifer for utt2num_frames')
  parser.add_argument(
    '--segments', type=str,
    help='segments-file format: each line is either'
         '<segment-id> <recording-id> <start-time> <end-time>'
         'e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5')
  parser.add_argument('rspecifier', type=str, help='WAV scp file')
  parser.add_argument('wspecifier', type=str, help='Write specifier')
  return parser

def compute_pitch():
  parser = get_parser()
  args = parser.parse_args()
  with kaldiio.ReadHelper(args.rspecifier, segments=args.segments) as reader, \
      init_kaldi_writer(args.wspecifier, write_num_frames=args.write_num_frames, compress=False) as writer:
    for utt_id, array in reader:
      array = array.astype(np.float32)
      config = {'window_length': args.window_length,
                'frame_length': args.frame_length,
                'thres_autoc': args.thres_autoc}
      pitch = Pitch.params(config).instantiate()
      pitch_feat = pitch(array, sample_rate=16000)
      writer[utt_id] = pitch_feat

if __name__ == '__main__':
  compute_pitch()





