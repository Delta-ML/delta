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
''' metrics utils of numpy '''
import numpy as np


#pylint: disable=invalid-name
def f1_score(tn, fp, fn, tp):
  '''  f1 score from confusion matrix '''
  del tn
  return 2 * tp / (2 * tp + fp + fn)


#pylint: disable=invalid-name
def stats_confusion(confusion):
  ''' confusion matrix to TP, FP, TN, FN '''
  FP = confusion.sum(axis=0) - np.diag(confusion)
  FN = confusion.sum(axis=1) - np.diag(confusion)
  TP = np.diag(confusion)
  TN = confusion.sum() - (TP + FN + TP)
  return TN, FP, FN, TP


def token_error(predict_seq_list=None, target_seq_list=None, eos_id=None):
  ''' computing token error
    :param predict_seq_list: (shape=[B,T], type=int), the list of predict token sequence
    :param target_seq_list: ([B,T], type=int), the list of target token sequence
    :param eos_id: (type=int), the end symbol of the target token sequence
    return: (type=float), the token error between predict and target token sequences
    '''
  if len(predict_seq_list) != len(target_seq_list):
    raise ValueError(
        'the number of prdict sequence and target sequence is not equal!')
  if eos_id is None:
    raise ValueError('the end symbol of target sequence is None!')

  levenshtein_distance, target_length = 0.0, 0.0

  for index, cur_predict_seq in enumerate(predict_seq_list):
    cur_target_seq = target_seq_list[index]
    if eos_id in cur_target_seq:
      target_seq_end = cur_target_seq.index(eos_id)
      cur_target_seq = cur_target_seq[:target_seq_end]
    if eos_id in cur_predict_seq:
      predict_seq_end = cur_predict_seq.index(eos_id)
      cur_predict_seq = cur_predict_seq[:predict_seq_end]

    levenshtein_distance += levenshtein(cur_predict_seq, cur_target_seq)
    target_length += len(cur_target_seq)

  errs = levenshtein_distance / target_length
  return errs


def levenshtein(short_seq, long_seq):
  ''' levenshtein distance
    :param short_seq: (shape=[T], type=int), the shorter sequence
    :param long_seq: (shape=[T], tpye=int), the longer sequence
    return: (type=float), the levenshtein distance between short_seq and long_seq
    '''
  min_len, max_len = len(short_seq), len(long_seq)
  if min_len > max_len:
    short_seq, long_seq = long_seq, short_seq
    min_len, max_len = max_len, min_len

  current = list(range(min_len + 1))
  for index in range(1, max_len + 1):
    previous, current = current, [index] + [0] * min_len
    for index_1 in range(1, min_len + 1):
      add, delete = previous[index_1] + 1, current[index_1 - 1] + 1
      change = previous[index_1 - 1]
      # string index from zero, but editditance from one
      if short_seq[index_1 - 1] != long_seq[index - 1]:
        change += 1
      current[index_1] = min(change, add, delete)
  return current[min_len]
