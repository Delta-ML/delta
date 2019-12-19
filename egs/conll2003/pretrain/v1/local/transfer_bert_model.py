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
import os
import sys
from bert import modeling
from absl import logging

import delta.compat as tf

def transfer_bert_model(bert_model_dir, output_bert_model):
  graph = tf.Graph()
  max_seq_len = 512
  num_labels = 2
  use_one_hot_embeddings = False
  with graph.as_default():
    with tf.Session() as sess:
      input_ids = tf.placeholder(tf.int32, (None, None), 'input_ids')
      input_mask = tf.placeholder(tf.int32, (None, None), 'input_mask')
      segment_ids = tf.placeholder(tf.int32, (None, None), 'segment_ids')

      bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_model_dir, 'bert_config.json'))
      model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
      all_encoder_layers = model.get_all_encoder_layers()
      input_x_bert_cls = model.get_pooled_output()
      for idx, layer in enumerate(all_encoder_layers):
        layer = tf.identity(layer, "encoder_layers_" + str(idx))
        print("layer:", layer)
      input_x_bert_cls = tf.identity(input_x_bert_cls, "input_x_bert_cls")
      print("input_x_bert_cls", input_x_bert_cls)
      saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, bert_model_dir + "/bert_model.ckpt")
      saver.save(sess, output_bert_model)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)

  if len(sys.argv) != 3:
    logging.error("Usage python {} bert_model_dir output_bert_model".format(sys.argv[0]))
    sys.exit(-1)
  bert_model_dir = sys.argv[1]
  output_bert_model = sys.argv[2]
  transfer_bert_model(bert_model_dir, output_bert_model)
