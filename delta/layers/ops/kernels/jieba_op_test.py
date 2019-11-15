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
''' jieba op test '''
import os
import time
import tempfile
import tensorflow as tf
from absl import logging

from delta import utils
from delta.layers.ops import py_x_ops

#pylint: disable=not-context-manager, invalid-name


def test_one(sess, ops, inputs):
  ''' elapse time of op '''
  t1 = time.time()
  sentence_out = sess.run(ops, inputs)
  t2 = time.time()
  logging.info("inputs: {}".format(inputs))
  logging.info("time cost: {}".format(t2 - t1))
  logging.info("\n".join([one_sen.decode("utf-8") for one_sen in sentence_out]))
  return sentence_out


class JiebaOpsTest(tf.test.TestCase):
  ''' jieba op test'''

  #pylint: disable=no-self-use
  def build_op(self, sentence):
    ''' build graph '''
    main_root = os.environ["MAIN_ROOT"]
    dict_path = os.path.join(main_root, "tools/cppjieba/dict/jieba.dict.utf8")
    hmm_path = os.path.join(main_root, "tools/cppjieba/dict/hmm_model.utf8")
    user_dict_path = os.path.join(main_root,
                                  "tools/cppjieba/dict/user.dict.utf8")
    idf_path = os.path.join(main_root, "tools/cppjieba/dict/idf.utf8")
    stop_word_path = os.path.join(main_root,
                                  "tools/cppjieba/dict/stop_words.utf8")

    words = py_x_ops.jieba_cut(
        sentence,
        hmm=True,
        dict_path=dict_path,
        hmm_path=hmm_path,
        user_dict_path=user_dict_path,
        idf_path=idf_path,
        stop_word_path=stop_word_path)
    return words

  def setUp(self):
    ''' set up '''
    self.graph_pb_path = tempfile.mktemp('jibeba_test.pb')
    logging.info("graph_pb_path: {}".format(self.graph_pb_path))
    graph = tf.Graph()
    with graph.as_default():
      sentence = tf.placeholder(
          dtype=tf.string, shape=[None], name="sentence_in")

      sentence_out = self.build_op(sentence)
      sentence_out = tf.identity(sentence_out, name="sentence_out")

      sess_config = utils.get_sess_config()
      sess = utils.get_session(sess_config)
      sess.run(tf.global_variables_initializer())

      utils.frozen_graph_to_pb(['sentence_out'], self.graph_pb_path, sess,
                               graph)

  def test_jieba_cut_op(self):
    ''' test jieba '''
    graph = tf.Graph()
    with graph.as_default():
      sentence_in = tf.placeholder(
          dtype=tf.string, shape=[None], name="sentence_in")

      sentence_out = self.build_op(sentence_in)

      with self.session(use_gpu=False) as sess:
        # self.assertShapeEqual(tf.shape(sentence_in), tf.shape(sentence_out))
        sentence_out_res = test_one(sess, sentence_out,
                                    {sentence_in: ["我爱北京天安门"]})
        self.assertEqual("我 爱 北京 天安门", sentence_out_res[0].decode("utf-8"))
        sentence_out_res = test_one(sess, sentence_out,
                                    {sentence_in: ["吉林省长春药店"]})
        self.assertEqual("吉林省 长春 药店", sentence_out_res[0].decode("utf-8"))
        sentence_out_res = test_one(sess, sentence_out,
                                    {sentence_in: ["吉林省长春药店", "南京市长江大桥"]})
        self.assertEqual(
            "吉林省 长春 药店\n南京市 长江大桥",
            "\n".join([one_sen.decode("utf-8") for one_sen in sentence_out_res
                      ]))

  def test_jibebaop_save_load(self):
    ''' test save jieba op '''
    sess_config = utils.get_sess_config()
    graph, sess = utils.load_graph_session_from_pb(
        self.graph_pb_path, sess_config, print_op=True)

    input_sentence_tensor = graph.get_operation_by_name(
        "sentence_in").outputs[0]
    sentence_out_output_tensor = graph.get_operation_by_name(
        "sentence_out").outputs[0]

    sentence_out_res = sess.run(sentence_out_output_tensor,
                                {input_sentence_tensor: ["南京市长江大桥"]})
    self.assertEqual("南京市 长江大桥", sentence_out_res[0].decode("utf-8"))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
