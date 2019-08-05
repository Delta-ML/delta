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
""" Split Kaldi data directory into traininng and validation sets. """
import argparse
from absl import logging

from delta.utils.kaldi import kaldi_dir
from delta.utils.kaldi import kaldi_dir_utils


def main():
  ''' The main function. '''
  logging.set_verbosity(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--num-spk-cv', type=float, default=0)
  parser.add_argument('--num-utt-cv', type=float, default=0)
  parser.add_argument('--cv-spk-percent', type=float, default=0.0)
  parser.add_argument('--cv-utt-percent', type=float, default=0.0)
  parser.add_argument('--fair-choice', type=bool, default=True)
  parser.add_argument('data_dir')
  parser.add_argument('data_dir_tr')
  parser.add_argument('data_dir_cv')

  args = parser.parse_args()

  num_spk_cv = args.num_spk_cv
  num_utt_cv = args.num_utt_cv
  if args.cv_spk_percent > 0:
    if args.cv_spk_percent >= 100:
      raise ValueError('cv_spk_percent cannot >= 100')
    num_spk_cv = args.cv_spk_percent / 100
  if args.cv_utt_percent > 0:
    if args.cv_utt_percent >= 100:
      raise ValueError('cv_utt_percent cannot >= 100')
    num_utt_cv = args.cv_utt_percent / 100
  if num_spk_cv == 0 and num_utt_cv == 0:
    num_spk_cv = 0.1

  meta = kaldi_dir.KaldiMetaData()
  meta.load(args.data_dir)
  meta_tr, meta_cv = kaldi_dir_utils.subset_data_dir_tr_cv(
      meta,
      num_spk_cv=num_spk_cv,
      num_utt_cv=num_utt_cv,
      fair_choice=args.fair_choice)
  logging.info('#spks tr: %d, cv: %d; #utts tr: %d, cv: %d' % (len(
      meta_tr.spks), len(meta_cv.spks), len(meta_tr.utts), len(meta_cv.utts)))

  meta_tr.dump(args.data_dir_tr, overwrite=True)
  meta_cv.dump(args.data_dir_cv, overwrite=True)


if __name__ == '__main__':
  main()
