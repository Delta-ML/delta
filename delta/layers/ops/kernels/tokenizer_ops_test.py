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
"""Tests for tokenizer_ops."""
import time
import tempfile
import delta.compat as tf
from absl import logging

from delta.layers.ops import py_x_ops


class TokenizerOpsTest(tf.test.TestCase):
  ''' tokenizer op test'''

  def setUp(self):
    super().setUp()
    self.vocab = [
        '</s>',
        '<unk>',
        'hello',
        '你好',
        'world',
    ]
    self.vocab_filepath = tempfile.mktemp(suffix='vocab.txt')
    with open(self.vocab_filepath, mode='w', encoding='utf-8') as fobj:
      for token in self.vocab:
        fobj.write(token)
        fobj.write('\n')

  def test_text_to_tokenid_with_vocab_file(self):
    ''' test label to token id'''
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # test batch
      start = time.time()
      batch_op = py_x_ops.sentence_to_ids(
          ['hello world', '你好 hello unknown  world'],
          maxlen=10,
          use_vocab_file=True,
          vocab_filepath=self.vocab_filepath,
          load_token_ids_from_vocab=False,
          pad_id=-1)
      token_ids, paddings = sess.run(batch_op)
      elapsed = time.time() - start
      logging.info("Time cost: {:.4f}s".format(elapsed))
      logging.info(token_ids)
      logging.info(paddings)
      logging.info("batch_op: {}".format(batch_op))
      self.assertAllEqual(token_ids, [[2, 4, -1, -1, -1, -1, -1, -1, -1, -1],
                                      [3, 2, 1, 4, -1, -1, -1, -1, -1, -1]])
      self.assertAllEqual(
          paddings,
          [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

      # test single
      single_op = py_x_ops.sentence_to_ids(
          '你好 hello unknown  world',
          maxlen=10,
          vocab_filepath=self.vocab_filepath,
          use_vocab_file=True,
          load_token_ids_from_vocab=False,
          pad_id=-1)
      token_ids, paddings = sess.run(single_op)
      logging.info("single_op: {}".format(single_op))
      self.assertAllEqual(token_ids, [3, 2, 1, 4, -1, -1, -1, -1, -1, -1])

      # test short single
      short_single_op = py_x_ops.sentence_to_ids(
          '你好 hello unknown  world',
          maxlen=2,
          use_vocab_file=True,
          vocab_filepath=self.vocab_filepath,
          load_token_ids_from_vocab=False,
          pad_id=0)
      token_ids, paddings = sess.run(short_single_op)
      logging.info("short_op: {}".format(short_single_op))
      self.assertAllEqual(token_ids, [3, 2])

      # test short batch
      short_batch_op = py_x_ops.sentence_to_ids(
          ['hello world', '你好 hello unknown  world'],
          maxlen=2,
          use_vocab_file=True,
          vocab_filepath=self.vocab_filepath,
          load_token_ids_from_vocab=False,
          pad_id=0)
      token_ids, paddings = sess.run(short_batch_op)
      logging.info("short_op: {}".format(short_batch_op))
      self.assertAllEqual(token_ids, [[2, 4], [3, 2]])

  def test_text_to_tokenid(self):
    ''' test label to token id'''
    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      # test batch
      start = time.time()
      batch_op, batch_padding_op = py_x_ops.sentence_to_ids(
          ['hello world', '你好 hello unknown  world'],
          maxlen=10,
          use_vocab_file=False,
          vocab=self.vocab,
          load_token_ids_from_vocab=False,
          pad_id=-1)
      batch_shape_op = tf.shape(batch_op)
      shape_res, token_ids, paddings = sess.run([batch_shape_op, batch_op, batch_padding_op])
      elapsed = time.time() - start
      logging.info("Time cost: {:.4f}s".format(elapsed))
      logging.info(token_ids)
      logging.info(paddings)
      logging.info("batch_op: {}".format(batch_op))
      logging.info(f"batch_shape: {shape_res}")
      self.assertAllEqual(shape_res, [2, 10])
      self.assertAllEqual(token_ids, [[2, 4, -1, -1, -1, -1, -1, -1, -1, -1],
                                      [3, 2, 1, 4, -1, -1, -1, -1, -1, -1]])
      self.assertAllEqual(
          paddings,
          [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

      # test single
      single_op, single_padding_op = py_x_ops.sentence_to_ids(
          '你好 hello unknown  world',
          maxlen=10,
          vocab=self.vocab,
          use_vocab_file=False,
          load_token_ids_from_vocab=False,
          pad_id=-1)
      single_shape_op = tf.shape(single_op)
      single_shape_res, token_ids, paddings = sess.run([single_shape_op, single_op, single_padding_op])
      logging.info("single_op: {}".format(single_op))
      logging.info(f"single_shape: {single_shape_res}")
      self.assertAllEqual(single_shape_res, [10])
      self.assertAllEqual(token_ids, [3, 2, 1, 4, -1, -1, -1, -1, -1, -1])

      # test short single
      short_single_op = py_x_ops.sentence_to_ids(
          '你好 hello unknown  world',
          maxlen=2,
          use_vocab_file=False,
          vocab=self.vocab,
          load_token_ids_from_vocab=False,
          pad_id=0)
      token_ids, paddings = sess.run(short_single_op)
      logging.info("short_op: {}".format(short_single_op))
      self.assertAllEqual(token_ids, [3, 2])

      # test short batch
      short_batch_op = py_x_ops.sentence_to_ids(
          ['hello world', '你好 hello unknown  world'],
          maxlen=2,
          use_vocab_file=False,
          vocab=self.vocab,
          load_token_ids_from_vocab=False,
          pad_id=0)
      token_ids, paddings = sess.run(short_batch_op)
      logging.info("short_op: {}".format(short_batch_op))
      self.assertAllEqual(token_ids, [[2, 4], [3, 2]])


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
