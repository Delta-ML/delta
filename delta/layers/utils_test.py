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
"""Test for layer utilities."""

from absl import logging
import delta.compat as tf

from delta.layers import utils
from delta.layers.utils import cut_or_padding
from delta.layers.utils import compute_sen_lens
from delta.layers.utils import compute_doc_lens
from delta.layers.utils import split_one_doc_to_true_len_sens

# pylint: disable=missing-docstring


class LayerUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

  def tearDown(self):
    ''' tear down'''

  def test_gelu(self):
    ''' test gelue activation '''

    # pylint: disable=invalid-name
    y = utils.gelu(tf.constant([0.5, 0.2], dtype=tf.float32))
    y_true = [0.345714, 0.11585142]

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      y_pred = sess.run(y)
      self.assertAllClose(y_pred, y_true)

  def test_cut_or_padding(self):
    # test for 1d
    origin_1_t = tf.placeholder(dtype=tf.int32, shape=[None])
    after_1_t = cut_or_padding(origin_1_t, 3)

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # test for padding
      res = sess.run(after_1_t, feed_dict={origin_1_t: [1, 2]})
      self.assertAllEqual(res, [1, 2, 0])

      # test for cut
      res = sess.run(after_1_t, feed_dict={origin_1_t: [1, 2, 3, 4, 5]})
      self.assertAllEqual(res, [1, 2, 3])

    # test for 2d
    origin_2_t = tf.placeholder(dtype=tf.int32, shape=[None, None])
    after_2_t = cut_or_padding(origin_2_t, 3)
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # test for padding
      res = sess.run(after_2_t, feed_dict={origin_2_t: [[1, 2], [1, 2]]})
      self.assertAllEqual(res, [[1, 2, 0], [1, 2, 0]])

      # test for cut
      res = sess.run(
          after_2_t, feed_dict={origin_2_t: [[1, 2, 3, 4], [1, 2, 3, 4]]})
      self.assertAllEqual(res, [[1, 2, 3], [1, 2, 3]])

  def test_compute_sen_lens(self):
    sentences = tf.placeholder(dtype=tf.int32)
    lens = compute_sen_lens(sentences)

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # test for 1d
      res = sess.run(lens, feed_dict={sentences: [1, 2, 0, 0]})
      self.assertEqual(res, 2)

      # test for 2d
      res = sess.run(lens, feed_dict={sentences: [[1, 2, 0, 0], [1, 2, 3, 4]]})
      self.assertAllEqual(res, [2, 4])

      # test for 3d
      res = sess.run(
          lens,
          feed_dict={
              sentences: [[[1, 2, 0, 0]], [[1, 2, 3, 4]], [[1, 0, 0, 0]]]
          })
      self.assertAllEqual(res, [[2], [4], [1]])

  def test_compute_doc_lens(self):
    ''' compute document length'''
    docs = tf.placeholder(dtype=tf.int32)
    lens = compute_doc_lens(docs)

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # test for 1d
      res = sess.run(lens, feed_dict={docs: [1, 2, 0, 0]})
      self.assertEqual(res, 2)

      # test for 2d
      res = sess.run(lens, feed_dict={docs: [[1, 2, 0, 0], [1, 2, 3, 4]]})
      self.assertAllEqual(res, [2, 4])

  def test_split_one_doc_to_true_len_sens(self):
    doc = tf.placeholder(dtype=tf.int32, shape=[None])
    split_token = 1
    padding_token = 0
    max_doc_len = 4
    max_sen_len = 5
    lens = split_one_doc_to_true_len_sens(doc, split_token, padding_token,
                                          max_doc_len, max_sen_len)

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      res = sess.run(lens, feed_dict={doc: [2, 3, 1, 2, 1, 2, 3, 4, 5, 6, 1]})
      self.assertAllEqual(
          res,
          [[2, 3, 0, 0, 0], [2, 0, 0, 0, 0], [2, 3, 4, 5, 6], [0, 0, 0, 0, 0]])

      all_empty = [[0 for _ in range(max_sen_len)] for _ in range(max_doc_len)]
      res = sess.run(lens, feed_dict={doc: []})
      self.assertAllEqual(res, all_empty)

      res = sess.run(lens, feed_dict={doc: [1, 1, 1, 1, 1]})
      self.assertAllEqual(res, all_empty)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
