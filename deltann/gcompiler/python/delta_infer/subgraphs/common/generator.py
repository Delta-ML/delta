from __future__ import absolute_import, division, print_function

import os
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.compat.v1 import Session

from .summarize_graph import GraphSummary

__all__ = ["RegistPattern"]


class RegistPattern(object):
  """ Rgist Pattern Decorator"""
  # a pattern map form name to list of GraphDef
  # note that a key name maybe map to multi graphdefs.
  patterns = {}

  def __init__(self, name=None):
    self.name = name
    if name not in RegistPattern.patterns:
      RegistPattern.patterns[name] = []

  @staticmethod
  def get_patterns(name):
    return RegistPattern.patterns[name]

  @staticmethod
  def Patterns():
    return RegistPattern.patterns

  def __call__(self, func):

    def local(*args, **kwargs):
      with Session() as sess:
        ret = func(*args, **kwargs)
        #sess.run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.global_variables_initializer().run()
        graph_summary = GraphSummary(graph_def=sess.graph_def)
        graph_summary.Summary()
        graph_def = graph_util.\
                convert_variables_to_constants(sess, sess.graph_def, graph_summary["outputs"])
        RegistPattern.patterns[self.name].append(graph_def)
      return ret

    return local
