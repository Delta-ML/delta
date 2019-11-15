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
""" Meta info for Kaldi data directory. """
import os
from random import Random
from absl import logging
import numpy as np

import kaldiio

from delta.utils.kaldi import kaldi_dir


def gen_dummy_meta(num_spk, num_utt_per_spk):
  ''' Generate a dummy data. '''
  meta = kaldi_dir.KaldiMetaData()
  for spk_idx in range(num_spk):
    for utt_idx in range(num_utt_per_spk):
      spk = str(spk_idx)
      utt = '%s_%d' % (spk, utt_idx)
      utt_meta = kaldi_dir.Utt()
      utt_meta.feat = 'foo/bar/feat/%s' % (utt)
      utt_meta.vad = 'foo/bar/vad/%s' % (utt)
      utt_meta.spk = spk
      meta.utts[utt] = utt_meta
  meta.collect_spks_from_utts()
  return meta


# pylint: disable=too-many-locals
def gen_dummy_data_dir(data_dir,
                       num_spk,
                       num_utt_per_spk,
                       feat_len=100,
                       feat_dim=40):
  ''' Generate a dummy data directory and return its meta. '''
  os.makedirs(data_dir, exist_ok=True)

  meta = kaldi_dir.KaldiMetaData()
  feats = {}
  vads = {}
  for spk_idx in range(num_spk):
    for utt_idx in range(num_utt_per_spk):
      spk = str(spk_idx)
      utt = '%s_%d' % (spk, utt_idx)
      utt_meta = kaldi_dir.Utt()
      feat_mat = np.ones((feat_len, feat_dim), dtype='float32')
      feats[utt] = feat_mat
      utt_meta.featlen = feat_len
      vad_mat = np.ones((feat_len,), dtype='float32')
      vads[utt] = vad_mat
      utt_meta.spk = spk
      meta.utts[utt] = utt_meta
  meta.collect_spks_from_utts()
  meta.dump(data_dir, True)

  feats_ark_path = os.path.join(data_dir, 'feats.ark')
  feats_scp_path = os.path.join(data_dir, 'feats.scp')
  kaldiio.save_ark(feats_ark_path, feats, scp=feats_scp_path, text=True)
  vad_ark_path = os.path.join(data_dir, 'vad.ark')
  vad_scp_path = os.path.join(data_dir, 'vad.scp')
  kaldiio.save_ark(vad_ark_path, vads, scp=vad_scp_path, text=True)

  loaded_meta = kaldi_dir.KaldiMetaData()
  loaded_meta.load(data_dir)
  return loaded_meta


def subset_data_dir_tr_cv(meta,
                          num_spk_cv=0,
                          num_utt_cv=0,
                          fair_choice=True,
                          keep_spk_id=True,
                          seed=123):
  '''
  Randomly split a data directory into training and validation set.
  This function acts like Kaldi's subset_data_dir_tr_cv.sh .

  Args:
    meta: a KaldiMetaData instance.
    num_spk_cv: number of speakers in the validation set. <1 means fraction.
                e.g. 0.1 = 10% of total speakers.
    num_utt_cv: number of utts in the validation set. <1 means fraction.
    fair_choice: utt mode only, choose utts from each spk with uniform chance.
    keep_spk_id: keep spk2id mapping from source data dir?
    seed: random seed.

  Returns:
    A tuple of (tr, cv) containing KaldiMetaData instances.

  Note:
    This function is definitive given the same seed.
  '''
  if num_spk_cv < 0 or num_utt_cv < 0:
    raise ValueError
  if num_spk_cv > 0 and num_utt_cv > 0:
    raise ValueError
  num_spks = len(meta.spks)
  num_utts = len(meta.utts)
  meta_tr = None
  meta_cv = None
  rand = Random(seed)

  if num_spk_cv > 0:
    if num_spk_cv < 1:
      num_spk_cv = int(num_spks * num_spk_cv)
    else:
      if num_spk_cv != int(num_spk_cv):
        raise ValueError
      num_spk_cv = int(num_spk_cv)
    logging.info('Intended #spks cv: %d' % (num_spk_cv))
    spks_shuffled = list(meta.spks.keys())
    rand.shuffle(spks_shuffled)
    spks_tr = spks_shuffled[num_spk_cv:]
    spks_cv = spks_shuffled[0:num_spk_cv]
    logging.info('Actual #spks cv: %d, tr: %d' % (len(spks_cv), len(spks_tr)))
    meta_tr = meta.select_spks(spks_tr)
    meta_cv = meta.select_spks(spks_cv)
    assert len(meta_tr.spks) + len(meta_cv.spks) == num_spks
  else:
    if num_utt_cv < 1:
      num_utt_cv = int(num_utts * num_utt_cv)
    else:
      if num_utt_cv != int(num_utt_cv):
        raise ValueError
      num_utt_cv = int(num_utt_cv)
    logging.info('Intended #utt cv: %d' % (num_utt_cv))
    if num_utt_cv > num_utts:
      num_utt_cv = num_utts

    logging.info('fair_choice = %s' % fair_choice)
    if fair_choice:
      logging.info('Applying fair choice across spks.')
      spk_utts = [list(spk.utts) for spk in meta.spks.values()]
      rand.shuffle(spk_utts)
      utts_tr = []
      utts_cv = []
      while len(utts_cv) < num_utt_cv:
        for utts in spk_utts:
          utt_idx = rand.randint(0, len(utts) - 1)
          utts_cv.append(utts.pop(utt_idx))
          if not utts:
            utts = []
            spk_utts.remove(utts)
          if len(utts_cv) >= num_utt_cv:
            break
      for utts in spk_utts:
        utts_tr.extend(utts)
    else:
      utts_shuffled = list(meta.utts.keys())
      rand.shuffle(utts_shuffled)
      utts_tr = utts_shuffled[num_utt_cv:]
      utts_cv = utts_shuffled[0:num_utt_cv]

    logging.info('Actual #utts cv: %d, tr: %d' % (len(utts_cv), len(utts_tr)))
    meta_tr = meta.select_utts(utts_tr)
    meta_cv = meta.select_utts(utts_cv)

  if keep_spk_id:
    for spk in meta.spks:
      meta_tr.spks[spk].id = meta.spks[spk].id
      meta_cv.spks[spk].id = meta.spks[spk].id

  assert len(meta_tr.utts) + len(meta_cv.utts) == num_utts
  return meta_tr, meta_cv
