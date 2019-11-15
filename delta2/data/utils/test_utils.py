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
"""Tests for data utilities."""
import json
import tempfile
import random
from pathlib import Path
import kaldiio
import numpy as np
from delta.data.utils.common_utils import JsonNumpyEncoder


def generate_json_data(config, mode, nexamples):
  """Generate Json data for test."""

  # pylint: disable=too-many-locals
  tmpdir = Path(tempfile.mkdtemp())
  ark = str(tmpdir.joinpath('test.ark'))
  scp = str(tmpdir.joinpath('test.scp'))
  ilens = 100
  nfeat = 40
  nexamples = nexamples
  desire_xs = []
  desire_ilens = []
  desire_ys = []
  desire_olens = []
  with kaldiio.WriteHelper('ark,scp:{},{}'.format(ark, scp)) as out_f:
    for i in range(nexamples):
      # pylint: disable=invalid-name
      x = np.random.random((ilens, nfeat)).astype(np.float32)
      uttid = 'uttid{}'.format(i)
      out_f[uttid] = x
      desire_xs.append(x)
      desire_ilens.append(ilens)
      desire_ys.append(np.array([1, 2, 3, 10]))
      desire_olens.append(4)

  dummy_json = {}
  dummy_json['utts'] = {}
  with open(scp, 'r') as out_f:
    for line in out_f:
      uttid, path = line.strip().split()
      dummy_json['utts'][uttid] = {
          'input': [{
              'feat': path,
              'name': 'input1',
              'shape': [ilens, nfeat]
          }],
          'output': [{
              'tokenid': '1 2 3 10',
              'name': 'output1',
              'shape': [4, 10]
          }]
      }

  path = tmpdir.joinpath('{}.json'.format(mode))
  path.touch(exist_ok=True)
  path = str(path.resolve())
  with open(path, 'w') as out_f:
    json.dump(dummy_json, out_f, cls=JsonNumpyEncoder)
    config['data'][mode]['paths'] = [path]

  return desire_xs, desire_ilens, desire_ys, desire_olens


def generate_vocab_file():
  """Generate Vocab file for test."""
  tmpdir = Path(tempfile.mkdtemp())
  vocab_file = str(tmpdir.joinpath('vocab.txt'))
  dummy_vocabs = ["</s>", "<unk>", "你好", "北京"]
  save_a_vocab_file(vocab_file, dummy_vocabs)
  return vocab_file


def save_a_vocab_file(vocab_file, vocab_list):
  """Save a Vocab file for test."""
  with open(vocab_file, "w", encoding='utf-8') as out_f:
    for vocab in vocab_list:
      out_f.write(vocab)
      out_f.write('\n')
  return vocab_file


def random_upsampling(data, sample_num):
  """Up sample"""
  np.random.seed(2019)
  new_indices = np.random.permutation(range(sample_num))
  original_size = len(data)
  new_data = [data[i % original_size] for i in new_indices]
  return new_data


def mock_a_text_file(sample_lines, line_num, file_name):
  """Generate a mock text file for test."""
  with open(file_name, "w", encoding="utf-8") as f:
    lines = random_upsampling(sample_lines, line_num)
    for line in lines:
      f.write(line + "\n")


def mock_a_npy_file(data, npy_name):
  """Generate a mock npy file for test."""
  data_arr = np.array(data)
  np.save(npy_name, data_arr)
