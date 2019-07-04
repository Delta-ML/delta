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

import sys
from absl import logging
from delta.data.utils.test_utils import mock_a_text_file

samples = ["1\t这是一部不错的电影。|我们都很喜欢。|也有教育意义。|",
           "0\t讲真的，这部电影不好看。|不是用太期待。|"]


def mock_text_class_data(train_file, dev_file, test_file):
  logging.info("Generate mock data: {}".format(train_file))
  mock_a_text_file(samples, 1000, train_file)
  logging.info("Generate mock data: {}".format(dev_file))
  mock_a_text_file(samples, 100, dev_file)
  logging.info("Generate mock data: {}".format(test_file))
  mock_a_text_file(samples, 100, test_file)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 4:
    logging.error("Usage python {} train_file, dev_file, test_file".format(sys.argv[0]))
    sys.exit(-1)

  train_file = sys.argv[1]
  dev_file = sys.argv[2]
  test_file = sys.argv[3]

  mock_text_class_data(train_file, dev_file, test_file)
