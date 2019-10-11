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

import sys
import tensorflow as tf
from absl import logging
from bilm.bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

def transfer_elmo_model(vocab_file, options_file, weight_file, token_embedding_file,
                        output_elmo_model):

  dump_token_embeddings(
      vocab_file, options_file, weight_file, token_embedding_file
  )
  logging.info("finish dump_token_embeddings")
  tf.reset_default_graph()

  with tf.Session(graph=tf.Graph()) as sess:
    bilm = BidirectionalLanguageModel(
      options_file,
      weight_file,
      use_character_inputs=False,
      embedding_weight_file=token_embedding_file
    )
    input_x = tf.placeholder(tf.int32, shape=[None, None],
                             name='input_x')
    train_embeddings_op = bilm(input_x)
    input_x_elmo_op = weight_layers(
      'output', train_embeddings_op, l2_coef=0.0
    )['weighted_op']
    input_x_elmo = tf.identity(input_x_elmo_op, name="input_x_elmo")
    logging.info("input_x_elmo shape: {}".format(input_x_elmo))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, output_elmo_model)
    logging.info("finish saving!")

    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in all_variables:
      logging.info("variable name: {}".format(v.name))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 6:
    logging.error("Usage python {} vocab_file, options_file, "
                  "weight_file, token_embedding_file, output_bert_model"
                  .format(sys.argv[0]))
    sys.exit(-1)

  vocab_file = sys.argv[1]
  options_file = sys.argv[2]
  weight_file = sys.argv[3]
  token_embedding_file = sys.argv[4]
  output_bert_model = sys.argv[5]
  transfer_elmo_model(vocab_file, options_file,
                      weight_file, token_embedding_file, output_bert_model)
