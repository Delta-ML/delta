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

''' inspect feature shape and dtype'''

import numpy as np
from absl import logging
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('path', None, 'feature path')
flags.DEFINE_boolean('verbose', False, 'dump data info')

def main(_):
  feat = np.load(FLAGS.path)
  logging.info(f"[{FLAGS.path}]")
  logging.info(f"  shape: {feat.shape}")
  logging.info(f"  dtype: {feat.dtype}")
  logging.info(f"  isnan: {np.all(np.isnan(feat))}")
  logging.info(f"  isinf: {np.all(np.isinf(feat))}")
  if FLAGS.verbose:
    logging.info(f"  data: {feat}")
    logging.info(f"  data: {feat[0][:]}")

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
