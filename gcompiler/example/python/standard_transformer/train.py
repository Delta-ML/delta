from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util
from model import *
from summarize_graph import GraphSummary


def create_model():
  batch_size = None
  seq_length = None
  hidden_size = 768
  num_attention_heads = 12
  size_per_head = int(hidden_size / num_attention_heads)

  layer_input = tf.placeholder(
      tf.float32, shape=(batch_size, seq_length, hidden_size))
  # Tensor of shape [batch_size, from_seq_length, to_seq_length].
  attention_mask = tf.placeholder(
      tf.float32, shape=(batch_size, seq_length, seq_length))

  #output_rnn = transformer_model(input_tensor=layer_input,
  #                               attention_mask=attention_mask,
  #                               hidden_size=hidden_size,
  #                               num_attention_heads=num_attention_heads,
  #                               intermediate_size=1280,
  #                               do_return_all_layers=False)

  out = transformer_model(
      input_tensor=layer_input,
      attention_mask=attention_mask,
      hidden_size=hidden_size,
      num_hidden_layers=3,
      num_attention_heads=num_attention_heads,
      intermediate_size=12802,
      intermediate_act_fn=gelu,
      hidden_dropout_prob=0.1,
      attention_probs_dropout_prob=0.1,
      initializer_range=0.02,
      do_return_all_layers=False)
  return out


def to_graph_def(graph_path):
  with tf.Session() as sess:
    ret = create_model()
    #sess.run(tf.compat.v1.global_variables_initializer())
    tf.global_variables_initializer().run()
    graph_summary = GraphSummary(graph_def=sess.graph_def)
    graph_summary.Summary()
    graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, graph_summary["outputs"])
    with open(graph_path, "wb") as f:
      f.write(graph_def.SerializeToString())


to_graph_def("./transformer_pattern.pb")
