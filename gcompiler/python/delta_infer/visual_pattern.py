from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import socket
import tempfile
import shutil

import netron
import tensorflow as tf
if tf.__version__ <= '1.8.0':
  from tensorflow import Session
  from tensorflow import summary
  from tensorflow import global_variables_initializer
else:
  from tensorflow.compat.v1 import Session
  from tensorflow.compat.v1 import summary
  from tensorflow.compat.v1 import global_variables_initializer
from tensorboard import program, default

from absl import app
from absl import flags

from .subgraphs import *

flags.DEFINE_enum('mode', 'visual',
                  ['visual', 'simplify', 'save_pattern', 'print_pattern'],
                  'running mode.')
flags.DEFINE_enum('type', 'netron', ['netron', 'tf'],
                  'running graph type of visual mode.')
flags.DEFINE_string('name', None, 'Pattern name.')
flags.DEFINE_integer(
    'idx', 0, "pattern graph index of RegistPattern.", lower_bound=0)

# used when in mode: visual and graph is from graph files.
flags.DEFINE_string('graph_path', None, 'tensorflow graph proto file.')
# used when in mode: visual
flags.DEFINE_string('pt', '8080', 'tensorboard or netron server port.')
# used when in save_pattern
flags.DEFINE_string('dir', None, 'save pattern to dir.')
# used when in mode: simplify
flags.DEFINE_string('outs', None, 'set outs of graph when in mode: simplify.')


def get_ip_address():
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  return s.getsockname()[0]


def visual_mode(mode_type='netron',
                pattern_name=None,
                pattern_idx=0,
                graph_path=None):

  def get_graph_def(pattern_name=None, pattern_idx=0, graph_path=None):
    assert (pattern_name is not None) or (graph_path is not None), \
            "pattern_name or graph_path should at least have one is None and the other is not."
    if graph_path is not None:
      name = graph_path.split('/')[-1]
      with open(graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    else:
      name = pattern_name
      graph_def = RegistPattern.get_patterns(pattern_name)[pattern_idx]
    return graph_def, name

  graph_def, name = get_graph_def(
      pattern_name=pattern_name, pattern_idx=pattern_idx, graph_path=graph_path)
  if mode_type == 'netron':
    tmp_dir = tempfile.mkdtemp()
    model_path = tmp_dir + "/" + name
    with open(model_path, "wb") as f:
      f.write(graph_def.SerializeToString())
    netron.start(file=model_path, host=get_ip_address(), port=flags.FLAGS.pt)
    shutil.rmtree(tmp_dir)

  else:  # type == 'tf'
    with Session() as sess:
      tmp_dir = tempfile.mkdtemp()
      tf.import_graph_def(graph_def)
      train_writer = summary.FileWriter(tmp_dir)
      train_writer.add_graph(sess.graph)
      train_writer.flush()
      train_writer.close()
      tb = program.TensorBoard(default.get_plugins())
      tb.configure(argv=[
          None, '--logdir', tmp_dir, '--port', flags.FLAGS.pt, '--host',
          get_ip_address()
      ])
      tb.main()
      shutil.rmtree(tmp_dir)


def main(argv):
  if flags.FLAGS.mode == "visual":
    visual_mode(
        mode_type=flags.FLAGS.type,
        pattern_name=flags.FLAGS.name,
        pattern_idx=flags.FLAGS.idx,
        graph_path=flags.FLAGS.graph_path)
  elif flags.FLAGS.mode == "save_pattern":
    if flags.FLAGS.name is None:
      raise ValueError("Flags 's pattern name({}) cann't be None.".format(
          flags.FLAGS.name))
    elif flags.FLAGS.dir == None:
      raise ValueError("Flags 's pattern dir({}) cann't be None.".format(
          flags.FLAGS.dir))
    else:
      with Session() as sess:
        graph_def = RegistPattern.get_patterns(
            flags.FLAGS.name)[flags.FLAGS.idx]
        model_path = flags.FLAGS.dir + "/" + flags.FLAGS.name + ".pb"
        with open(model_path, "wb") as f:
          f.write(graph_def.SerializeToString())
      sess.close()
  elif flags.FLAGS.mode == "simplify":
    if flags.FLAGS.graph_path is None:
      raise ValueError("Flags 's graph_path({}) cann't be None.".format(
          flags.FLAGS.graph_path))
    else:
      with Session() as sess:
        with open(flags.FLAGS.graph_path, "rb") as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
          g_in = tf.import_graph_def(graph_def)
          global_variables_initializer().run()
          #graph_summary = GraphSummary(graph_def=sess.graph_def)
          #graph_summary.Summary()
          outs = flags.FLAGS.outs.split(',')
          graph_def = graph_util.convert_variables_to_constants(
              sess, sess.graph_def, outs)
          graph_summary = GraphSummary(graph_def=graph_def)
          graph_summary.Summary()
          graph_def_name = flags.FLAGS.graph_path.split('/')[-1]
          model_path = flags.FLAGS.dir + "/" + graph_def_name
          with open(model_path, "wb") as f:
            f.write(graph_def.SerializeToString())
  else:
    for pattern_name in RegistPattern.Patterns().keys():
      print("Registered Pattern name: {} with {} different GraphDef.".\
              format(pattern_name, len(RegistPattern.get_patterns(pattern_name))))


def command():
  app.run(main)
