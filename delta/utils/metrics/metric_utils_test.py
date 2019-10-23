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
''' metrics utils unittest '''
import numpy as np
import delta.compat as tf

from delta.utils.metrics import metric_utils


class MetricUtilsTest(tf.test.TestCase):
  ''' metrics utils unittest'''

  def setUp(self):
    super().setUp()

  def tearDown(self):
    ''' tear down '''

  #pylint: disable=invalid-name
  def test_stats_confusion(self):
    ''' test stats of confusion matrix'''
    confusion = np.array([[5, 1, 1, 0], [0, 10, 0, 1], [0, 0, 6, 1],
                          [0, 0, 1, 6]])

    tn, fp, fn, tp = metric_utils.stats_confusion(confusion)
    self.assertEqual(len(tp), 4)
    self.assertAllEqual(tn, [20, 11, 19, 19])
    self.assertAllEqual(fp, [0, 1, 2, 2])
    self.assertAllEqual(fn, [2, 1, 1, 1])
    self.assertAllEqual(tp, [5, 10, 6, 6])

  #pylint: disable=invalid-name
  def test_f1_score(self):
    ''' test f1 score '''
    confusion = np.array([[5, 1, 1, 0], [0, 10, 0, 1], [0, 0, 6, 1],
                          [0, 0, 1, 6]])
    tn, fp, fn, tp = metric_utils.stats_confusion(confusion)

    f1 = metric_utils.f1_score(tn, fp, fn, tp)
    self.assertAllClose(f1, [0.83333333, 0.90909091, 0.8, 0.8])

  def test_token_error(self):
    ''' test token_error '''
    seq_list_one = [[5, 1, 1, 1, 1], [5, 2, 6, 10, 2]]
    seq_list_two = [[5, 2, 3, 1, 2], [5, 2, 6, 10, 2]]

    token_errors = metric_utils.token_error(
        seq_list_one, seq_list_two, eos_id=0)
    self.assertAllClose(token_errors, 0.3)

    seq_list_three = [[5, 1, 1, 1, 1, 0, 0], [5, 2, 6, 10, 2, 0]]
    token_errors = metric_utils.token_error(
        seq_list_three, seq_list_two, eos_id=0)
    self.assertAllClose(token_errors, 0.3)

    token_errors = metric_utils.token_error(
        seq_list_two, seq_list_three, eos_id=0)
    self.assertAllClose(token_errors, 0.3)

  def test_levenshtein(self):
    ''' test levenshtein distance '''
    seq_one = [5, 1, 1, 1, 0]
    seq_two = [5, 2, 3, 0, 2]

    levenshtein_distance = metric_utils.levenshtein(seq_one, seq_two)
    self.assertAllClose(levenshtein_distance, 4)

    #another case
    seq_one = [5, 1, 1, 1, 0]
    seq_two = [5, 1, 1, 1, 0]

    levenshtein_distance = metric_utils.levenshtein(seq_one, seq_two)
    self.assertAllClose(levenshtein_distance, 0)


if __name__ == "__main__":
  tf.test.main()
