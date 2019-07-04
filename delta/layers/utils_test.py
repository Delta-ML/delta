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

import os
import tempfile
from pathlib import Path
from absl import logging
import tensorflow as tf

from delta.layers.utils import cut_or_padding
from delta.layers.utils import compute_sen_lens
from delta.layers.utils import compute_doc_lens
from delta.layers.utils import split_one_doc_to_true_len_sens

# pylint: disable=missing-docstring


class LayerUtilsTest(tf.test.TestCase):

  def test_cut_or_padding(self):
    # test for 1d
    origin_1_t = tf.placeholder(dtype=tf.int32, shape=[None])
    after_1_t = cut_or_padding(origin_1_t, 3)

    with self.session() as sess:
      # test for padding
      res = sess.run(after_1_t, feed_dict={origin_1_t: [1, 2]})
      logging.info(res)
      self.assertAllEqual(res, [1, 2, 0])

      # test for cut
      res = sess.run(after_1_t, feed_dict={origin_1_t: [1, 2, 3, 4, 5]})
      logging.info(res)
      self.assertAllEqual(res, [1, 2, 3])

    # test for 2d
    origin_2_t = tf.placeholder(dtype=tf.int32, shape=[None, None])
    after_2_t = cut_or_padding(origin_2_t, 3)
    with self.session() as sess:
      # test for padding
      res = sess.run(after_2_t, feed_dict={origin_2_t: [[1, 2], [1, 2]]})
      logging.info(res)
      self.assertAllEqual(res, [[1, 2, 0], [1, 2, 0]])

      # test for cut
      res = sess.run(
          after_2_t, feed_dict={origin_2_t: [[1, 2, 3, 4], [1, 2, 3, 4]]})
      logging.info(res)
      self.assertAllEqual(res, [[1, 2, 3], [1, 2, 3]])

  def test_compute_sen_lens(self):
    sentences = tf.placeholder(dtype=tf.int32)
    lens = compute_sen_lens(sentences)

    with self.session() as sess:
      # test for 1d
      res = sess.run(lens, feed_dict={sentences: [1, 2, 0, 0]})
      logging.info(res)
      self.assertEqual(res, 2)

      # test for 2d
      res = sess.run(lens, feed_dict={sentences: [[1, 2, 0, 0], [1, 2, 3, 4]]})
      logging.info(res)
      self.assertAllEqual(res, [2, 4])

      # test for 3d
      res = sess.run(
          lens,
          feed_dict={
              sentences: [[[1, 2, 0, 0]], [[1, 2, 3, 4]], [[1, 0, 0, 0]]]
          })
      logging.info(res)
      self.assertAllEqual(res, [[2], [4], [1]])

  def test_compute_doc_lens(self):
    docs = tf.placeholder(dtype=tf.int32)
    lens = compute_doc_lens(docs)

    with self.session() as sess:
      # test for 1d
      res = sess.run(lens, feed_dict={docs: [1, 2, 0, 0]})
      logging.info(res)
      self.assertEqual(res, 2)

      # test for 2d
      res = sess.run(lens, feed_dict={docs: [[1, 2, 0, 0], [1, 2, 3, 4]]})
      logging.info(res)
      self.assertAllEqual(res, [2, 4])

  def test_split_one_doc_to_true_len_sens(self):
    doc = tf.placeholder(dtype=tf.int32, shape=[None])
    split_token = 1
    padding_token = 0
    max_doc_len = 4
    max_sen_len = 5
    lens = split_one_doc_to_true_len_sens(doc, split_token, padding_token,
                                          max_doc_len, max_sen_len)

    with self.session() as sess:
      res = sess.run(lens, feed_dict={doc: [2, 3, 1, 2, 1, 2, 3, 4, 5, 6, 1]})
      logging.info(res)
      self.assertAllEqual(
          res,
          [[2, 3, 0, 0, 0], [2, 0, 0, 0, 0], [2, 3, 4, 5, 6], [0, 0, 0, 0, 0]])

      all_empty = [[0 for _ in range(max_sen_len)] for _ in range(max_doc_len)]
      res = sess.run(lens, feed_dict={doc: []})
      logging.info(res)
      self.assertAllEqual(res, all_empty)

      res = sess.run(lens, feed_dict={doc: [1, 1, 1, 1, 1]})
      logging.info(res)
      self.assertAllEqual(res, all_empty)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
