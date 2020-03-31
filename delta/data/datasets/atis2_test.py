# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for compat.py."""

from absl import logging

import delta.compat as tf
from delta.data.datasets.atis2 import ATIS2


class ATIS2Test(tf.test.TestCase):
  """data class test for nlu-joint task."""

  def test_build(self):
    atis2 = ATIS2('/atis2')
    atis2.build()
    self.assertTrue(atis2.is_ready())


if __name__ == '__main__':
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
