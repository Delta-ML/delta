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
''' text ops utils unittest'''

# pylint: disable=missing-docstring

import delta.compat as tf
from absl import logging
import tempfile

from delta.data.preprocess.text_ops import clean_english_str_tf
from delta.data.preprocess.text_ops import char_cut_tf
from delta.data.preprocess.text_ops import tokenize_label
from delta.data.preprocess.text_ops import tokenize_sentence


class TextOpsTest(tf.test.TestCase):

  def setUp(self):
    ''' set up '''
    vocab_text = ['<unk>\t1', '</s>\t2', 'O\t3']
    vocab_label = [
        'B\t0', "B-PER\t1", "I-PER\t2", "B-LOC\t3", "I-LOC\t4", "B-ORG5\t5",
        "I-ORG\t6", "B-MISC\t7", "I-MISC\t8"
    ]
    self.vocab_text_filepath = tempfile.mktemp(suffix='text_vocab.txt')
    self.vocab_label_filepath = tempfile.mktemp(suffix='label_vocab.txt')
    with open(self.vocab_text_filepath, mode='w', encoding='utf-8') as fobj:
      for token in vocab_text:
        fobj.write(token)
        fobj.write('\n')
    with open(self.vocab_label_filepath, mode='w', encoding='utf-8') as fobj:
      for token in vocab_label:
        fobj.write(token)
        fobj.write('\n')

  def tearDown(self):
    ''' tear down '''

  def test_label_and_text(self):
    text = ["O O"]
    maxlen = 2
    text_tokenize_t = tokenize_sentence(text, maxlen, self.vocab_text_filepath)
    label = ["B B"]
    maxlen = 2
    label_tokenize_t = tokenize_label(label, maxlen, self.vocab_label_filepath,
                                      -1)

    with self.session() as sess:
      res = sess.run([text_tokenize_t, label_tokenize_t])
      logging.debug(res)
      self.assertAllEqual(res[0],
                          [[3, 3]]), self.assertAllEqual(res[1], [[0, 0]])

  def test_clean_english_str_tf(self):
    t_sentence_in = tf.placeholder(dtype=tf.string)
    t_sentence_out = clean_english_str_tf(t_sentence_in)
    with self.session(use_gpu=False) as sess:
      sentence_out = sess.run(t_sentence_out,
                              {t_sentence_in: "I'd like to have an APPLE! "})
      logging.info(sentence_out)
      self.assertEqual("i 'd like to have an apple !",
                       sentence_out.decode("utf-8"))
      sentence_out = sess.run(t_sentence_out,
                              {t_sentence_in: ["I'd like to have an APPLE! "]})
      logging.info(sentence_out)
      self.assertEqual("i 'd like to have an apple !",
                       sentence_out[0].decode("utf-8"))

  def test_char_cut_tf_str(self):
    t_sen_in = tf.placeholder(dtype=tf.string, shape=())
    t_sen_out = char_cut_tf(t_sen_in)
    with self.session(use_gpu=False) as sess:
      sen_out = sess.run(t_sen_out, {t_sen_in: "我爱北京天安门"})
      logging.info(sen_out.decode("utf-8"))
      self.assertEqual("我 爱 北 京 天 安 门", sen_out.decode("utf-8"))

  def test_char_cut_tf_list(self):
    t_sen_in = tf.placeholder(dtype=tf.string, shape=(None,))
    t_sen_out = char_cut_tf(t_sen_in)
    with self.session(use_gpu=False) as sess:
      sen_out = sess.run(t_sen_out, {t_sen_in: ["我爱北京天安门", "天安门前太阳升啊"]})
      logging.info([one.decode("utf-8") for one in sen_out])
      self.assertAllEqual(["我 爱 北 京 天 安 门", "天 安 门 前 太 阳 升 啊"],
                          [one.decode("utf-8") for one in sen_out])


if __name__ == '__main__':
  logging.set_verbosity(logging.DEBUG)
  tf.test.main()
