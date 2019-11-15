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
''' ngram op unittest'''
import delta.compat as tf
from absl import logging

from delta.layers.ops import py_x_ops


class NGramOpsTest(tf.test.TestCase):
  ''' ngram op test'''

  def setUp(self):
    super().setUp()
    self.testcase = [[0, 0, 0, 0], [223, 0, 0, 0], [0, 8, 0, 0], [4, 8, 0, 0],
                     [0, 0, 10, 0], [2, 5, 3, 0], [7, 2, 1, 24]]

  def test_ngram_op_2_order(self):
    ''' test ngram 2-order op'''
    ground_truth_2 = [0, 0, 0, 0, 0, 0, 0]

    word_ngram = 2
    t_input = tf.placeholder(shape=(4,), dtype=tf.int32)
    t_ngram = py_x_ops.ngram(
        t_input, word_ngrams=word_ngram, vocab_size=5000, bucket_size=100000)
    logging.info("t_ngram: {}".format(t_ngram))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      ngram_result = sess.run(t_ngram, feed_dict={t_input: self.testcase[0]})
      self.assertAllEqual(ngram_result, ground_truth_2)

  def test_batch_ngram_op_2_order(self):
    ''' tset batch 2-order ngram '''
    ground_truth_2 = [[0, 0, 0, 0, 0, 0, 0], [223, 0, 0, 0, 0, 0, 0],
                      [0, 8, 5008, 0, 0, 0, 0], [4, 8, 102492, 0, 0, 0, 0],
                      [0, 0, 10, 5000, 5010, 0, 0],
                      [2, 5, 3, 103747, 51858, 0, 0],
                      [7, 2, 1, 24, 50599, 103743, 54395]]

    word_ngram = 2
    t_input = tf.placeholder(shape=(7, 4), dtype=tf.int32)
    t_ngram = py_x_ops.ngram(
        t_input, word_ngrams=word_ngram, vocab_size=5000, bucket_size=100000)
    logging.info("batch t_ngram: {}".format(t_ngram))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      ngram_result = sess.run(t_ngram, feed_dict={t_input: self.testcase})
      ngram_result = [list(res) for res in ngram_result]
      self.assertAllEqual(ngram_result, ground_truth_2)

  def test_batch_ngram_op_3_order(self):
    ''' test batch 3-order ngram '''

    ground_truth_3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [223, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 8, 5008, 0, 0, 0, 0, 0, 0],
                      [4, 8, 102492, 0, 0, 0, 0, 0, 0],
                      [0, 0, 10, 5000, 5010, 5010, 0, 0, 0],
                      [2, 5, 3, 103747, 43140, 51858, 0, 0, 0],
                      [7, 2, 1, 24, 50599, 73230, 103743, 45677, 54395]]

    word_ngram = 3
    t_input = tf.placeholder(shape=(7, 4), dtype=tf.int32)
    t_ngram = py_x_ops.ngram(
        t_input, word_ngrams=word_ngram, vocab_size=5000, bucket_size=100000)
    logging.info("batch t_ngram: {}".format(t_ngram))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      ngram_result = sess.run(t_ngram, feed_dict={t_input: self.testcase})
      ngram_result = [list(res) for res in ngram_result]
      self.assertAllEqual(ngram_result, ground_truth_3)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
