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

''' compute cmvn '''

import numpy as np
from absl import logging
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('path_file', None, 'filelist of path')
flags.DEFINE_string('cmvn_path', None, 'cmvn path')

def main(_):
  counts = 0.0
  sum_feats = None
  square_sum_feats = None

  idx = 0
  with open(FLAGS.path_file, 'r') as fin:
    for idx, path in enumerate(fin, 1):
      path = path.strip()
      feat = np.load(path)
      if sum_feats is None and square_sum_feats is None:
         feat_shape = feat.shape[1:]
         logging.info(f"feat_shape: {feat_shape}")
         # Accumulate in double precision
         sum_feats = np.zeros(feat_shape, dtype=np.float64)
         square_sum_feats = np.zeros(feat_shape, dtype=np.float64)

      counts += feat.shape[0]
      sum_feats += feat.sum(axis=0)
      square_sum_feats += np.sum(np.power(feat, 2), axis=0)

  logging.info(f"sums: {sum_feats}")
  logging.info(f"square: {square_sum_feats}")
  logging.info(f"counts: {counts}")
  mean = sum_feats / counts
  var = square_sum_feats / counts - np.power(mean, 2)

  np.save(FLAGS.cmvn_path, (mean, var))

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
