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
''' python metric unittest '''
import os
from pathlib import Path
import tempfile
import numpy as np
import delta.compat as tf

from delta import utils
from delta.utils import metrics
from delta import PACKAGE_ROOT_DIR


#pylint: disable=too-many-instance-attributes
class MetricTest(tf.test.TestCase):
  ''' python metrix unittest '''

  def setUp(self):
    super().setUp()
    ''' setup '''
    package_root = Path(PACKAGE_ROOT_DIR)
    self.config_file_crf = \
      package_root.joinpath('../egs/mock_text_seq_label_data/seq-label/v1/config/seq-label-mock.yml')

    self.conf_str = '''
      solver:
        metrics:
          pos_label: 1 # int, same to sklearn
          cals:
          - name: AccuracyCal
            arguments: null 
          - name: ConfusionMatrixCal
            arguments: null
          - name: PrecisionCal
            arguments:
              average: 'micro'
          - name: RecallCal
            arguments:
              average: 'micro'
          - name: F1ScoreCal
            arguments:
              average: 'micro'
    '''

    self.conf_file = tempfile.mktemp(suffix='metric.yaml')
    with open(self.conf_file, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.conf_str)

    self.true_label = np.array([1, 1, 2, 3, 4, 6, 5])
    self.pred1 = np.array([1, 1, 2, 3, 4, 6, 5])
    self.pred2 = np.array([2, 2, 1, 1, 1, 1, 1])

    # config for test token error metircs
    self.token_conf_str = '''
      solver:
        metrics:
          pos_label: 1 # int, same to sklearn
          cals:
          - name: TokenErrCal
            arguments:
              eos_id: 0
    '''

    self.token_conf_file = tempfile.mktemp(suffix='token.yaml')
    with open(self.token_conf_file, 'w', encoding='utf-8') as f:  #pylint: disable=invalid-name
      f.write(self.token_conf_str)

    self.token_true_label = [[1, 1, 1, 1], [1, 3, 4, 5]]
    self.token_pred1 = [[1, 1, 1, 1], [1, 3, 4, 5]]
    self.token_pred2 = [[1, 2, 2, 2], [1, 0, 0, 0]]

  def tearDown(self):
    ''' tear down '''
    if os.path.exists(self.conf_file):
      os.unlink(self.conf_file)

  def test_metric(self):
    ''' test get_metrics function '''
    config = utils.load_config(self.conf_file)

    metrics1 = metrics.get_metrics(
        config, y_true=self.true_label, y_pred=self.pred1)
    self.assertEqual(1.0, metrics1['AccuracyCal'])
    self.assertEqual(1.0, metrics1['PrecisionCal'])
    self.assertEqual(1.0, metrics1['RecallCal'])
    self.assertEqual(1.0, metrics1['F1ScoreCal'])

    metrics2 = metrics.get_metrics(
        config, y_true=self.true_label, y_pred=self.pred2)
    self.assertEqual(0.0, metrics2['AccuracyCal'])
    self.assertEqual(0.0, metrics2['PrecisionCal'])
    self.assertEqual(0.0, metrics2['RecallCal'])
    self.assertEqual(0.0, metrics2['F1ScoreCal'])

  def test_token_err(self):
    ''' test tooken error rate '''
    config = utils.load_config(self.token_conf_file)

    metrics1 = metrics.get_metrics(
        config, y_true=self.token_true_label, y_pred=self.token_pred1)
    self.assertEqual(0.0, metrics1['TokenErrCal'])

    metrics2 = metrics.get_metrics(
        config, y_true=self.token_true_label, y_pred=self.token_pred2)
    self.assertEqual(0.75, metrics2['TokenErrCal'])

  def test_crf_metrics(self):
    ''' test crf metrics '''
    config = utils.load_config(self.config_file_crf)
    metrics3 = metrics.get_metrics(
        config, y_true=[self.true_label], y_pred=[self.pred1])
    # metrics3: one string. Text summary of the precision, recall, F1 score for each class.
    # res3 = metrics3['CrfCal']
    # print(res3)
    # for i, s in enumerate(res3):
    #   print(i, s)
    self.assertEqual('1.0000', metrics3['CrfCal'][67:73])

    metrics4 = metrics.get_metrics(
        config, y_true=[self.true_label], y_pred=[self.pred2])
    self.assertEqual('0.0000', metrics4['CrfCal'][67:73])


if __name__ == "__main__":
  tf.test.main()
