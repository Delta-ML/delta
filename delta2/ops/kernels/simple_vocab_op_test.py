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
"""Tests for simple_vocab."""
import delta.compat as tf
from delta.layers.ops import py_x_ops


class VocabOpsTest(tf.test.TestCase):
  ''' vocab op test '''

  def setUp(self):
    super().setUp()

  def tearDown(self):
    ''' tear down '''

  def test_vocab_token_to_id(self):
    ''' tset vocab token to id'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      vocab = [
          '<s>',
          '</s>',
          '<unk>',
          '<epsilon>',
          'a',
          'b c d e',
          'øut',
          'über',
          '♣',
          '愤青',
          '←',
      ]
      self.assertEqual(0, py_x_ops.vocab_token_to_id('<s>', vocab=vocab).eval())
      self.assertEqual(4, py_x_ops.vocab_token_to_id('a', vocab=vocab).eval())
      self.assertAllEqual([5, 8],
                          py_x_ops.vocab_token_to_id(['b c d e', '♣'],
                                                     vocab=vocab).eval())
      self.assertEqual(
          2,
          py_x_ops.vocab_token_to_id('unknown', vocab=vocab).eval())

  def test_vocab_token_to_load_id(self):
    ''' test vocab token to id which is loaded from vocab file'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      vocab = [
          '<s>	3',
          '</s>	5',
          '<unk>	7',
          '<epsilon>	9',
          'a	2',
          'b c d e	4',
          'øut	8',
          'über	10',
          '♣	-1',
          '愤青	-3',
          '←	-5',
      ]
      self.assertEqual(
          3,
          py_x_ops.vocab_token_to_id(
              '<s>', vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          2,
          py_x_ops.vocab_token_to_id(
              'a', vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertAllEqual([4, -1],
                          py_x_ops.vocab_token_to_id(
                              ['b c d e', '♣'],
                              vocab=vocab,
                              load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          7,
          py_x_ops.vocab_token_to_id(
              'unknown', vocab=vocab, load_token_ids_from_vocab=True).eval())

  def test_vocab_id_to_token(self):
    ''' test vocab id to token '''
    with self.cached_session(use_gpu=False, force_gpu=False):
      vocab = [
          '<s>',
          '</s>',
          '<unk>',
          '<epsilon>',
          'a',
          'b c d e',
          'øut',
          'über',
          '♣',
          '愤青',
          '←',
      ]
      self.assertEqual(
          '<s>',
          py_x_ops.vocab_id_to_token(0, vocab=vocab).eval().decode('utf-8'))
      self.assertEqual(
          'a',
          py_x_ops.vocab_id_to_token(4, vocab=vocab).eval().decode('utf-8'))

      res = py_x_ops.vocab_id_to_token([5, 8], vocab=vocab).eval()
      res = [r.decode('utf-8') for r in res]
      self.assertAllEqual(['b c d e', '♣'], res)
      self.assertEqual(
          '<unk>',
          py_x_ops.vocab_id_to_token(2, vocab=vocab).eval().decode('utf-8'))
      self.assertEqual(
          '<unk>',
          py_x_ops.vocab_id_to_token(-1, vocab=vocab).eval().decode('utf-8'))
      self.assertEqual(
          '<unk>',
          py_x_ops.vocab_id_to_token(11, vocab=vocab).eval().decode('utf-8'))

  def test_vocab_id_to_token_load_id(self):
    ''' test vocab id to token which is loaded from vocabfile'''
    with self.cached_session(use_gpu=False, force_gpu=False):
      vocab = [
          '<s>	3',
          '</s>	5',
          '<unk>	7',
          '<epsilon>	9',
          'a	2',
          'b c d e	4',
          'øut	8',
          'über	10',
          '♣	-1',
          '愤青	-3',
          '←	-5',
      ]
      self.assertEqual(
          '<s>',
          py_x_ops.vocab_id_to_token(
              3, vocab=vocab,
              load_token_ids_from_vocab=True).eval().decode('utf-8'))
      self.assertEqual(
          'a',
          py_x_ops.vocab_id_to_token(
              2, vocab=vocab,
              load_token_ids_from_vocab=True).eval().decode('utf-8'))
      res = py_x_ops.vocab_id_to_token([4, -1],
                                       vocab=vocab,
                                       load_token_ids_from_vocab=True).eval()
      res = [r.decode('utf-8') for r in res]

      self.assertAllEqual(['b c d e', '♣'], res)
      self.assertEqual(
          '<unk>',
          py_x_ops.vocab_id_to_token(
              7, vocab=vocab,
              load_token_ids_from_vocab=True).eval().decode('utf-8'))
      self.assertEqual(
          '<unk>',
          py_x_ops.vocab_id_to_token(
              0, vocab=vocab,
              load_token_ids_from_vocab=True).eval().decode('utf-8'))

  def test_token_in_vocab(self):
    '''test token whether in vocab '''
    with self.cached_session(use_gpu=False, force_gpu=False):
      vocab = [
          '<s>',
          '</s>',
          '<unk>',
          '<epsilon>',
          'a',
          'b c d e',
          'øut',
          'über',
          '♣',
          '愤青',
          '←',
      ]
      self.assertTrue(py_x_ops.token_in_vocab('a', vocab=vocab).eval())
      self.assertTrue(py_x_ops.token_in_vocab('<unk>', vocab=vocab).eval())
      self.assertTrue(
          py_x_ops.token_in_vocab(['b c d e', '♣'], vocab=vocab).eval().all())
      self.assertFalse(py_x_ops.token_in_vocab('unknown', vocab=vocab).eval())


if __name__ == '__main__':
  tf.test.main()
