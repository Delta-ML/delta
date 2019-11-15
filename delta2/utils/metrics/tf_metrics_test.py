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
''' tf metrics utils unittest '''
import delta.compat as tf

from delta.utils.metrics import tf_metrics


class TFMetricUtilsTest(tf.test.TestCase):
  ''' tf metrics utils unittest'''

  def test_accuracy(self):
    ''' test accuracy'''
    logits = tf.constant([[0.1, 0.2, 0.7], [0.5, 0.2, 0.3], [0.2, 0.2, 0.6]])
    labels = tf.constant([2, 0, 1])
    output = tf_metrics.accuracy(logits, labels)
    self.assertAllClose(output, 0.6666667)

  def test_confusion_matrix(self):
    '''test confusion matrix'''
    logits = tf.constant([[0.1, 0.2, 0.7], [0.5, 0.2, 0.3], [0.6, 0.1, 0.3],
                          [0.2, 0.3, 0.5], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]])
    labels = tf.constant([2, 0, 0, 2, 1, 1])
    num_class = 3
    output = tf_metrics.confusion_matrix(logits, labels, num_class)
    output_true = tf.constant([[2, 0, 0], [0, 1, 1], [0, 0, 2]])
    self.assertAllEqual(output, output_true)


if __name__ == "__main__":
  tf.test.main()
