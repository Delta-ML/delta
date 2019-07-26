# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' ctc python decoder '''

from itertools import groupby
import numpy as np


def ctc_greedy_decode(predict_seqs, blank_id, unique=True):
  """
        decod using simple greedy search
        :param predict_seqs: ([B, T, C], type=float) the output of CTC-based asr model.
        :param blank_id: (type=int) the index of blank symbol
        :param unique: (type=boolean) if merge consecutive repeated classes in output
        :return: ([B, T]) the result of greedy decode
        """
  decode_seq = []

  for predict_seq in predict_seqs:
    cur_token_list = [np.argmax(prob_list) for prob_list in predict_seq]

    cur_decode_list = []
    if unique:
      temp = groupby(cur_token_list)
      cur_decode_list = [key for key, group in temp if key != blank_id]
    else:
      cur_decode_list = [
          token_id for token_id in cur_token_list if token_id != blank_id
      ]
    decode_seq.append(cur_decode_list)

  return decode_seq
