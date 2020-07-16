from __future__ import absolute_import, division, print_function
import os
import sys

from .subgraphs import *
from .cpp import DeltaGraph, AutoOptimizer, RegisterPattern

__all__ = ["GraphStream"]


class GraphStream(object):
  """ GraphStream is a class for delta automatical optimization"""

  def __init__(self, pb_path):
    self.__path = pb_path
    self.__hint_map = {}

  def __run(self):
    graph_defs = []
    for pattern_name in RegistPattern.Patterns().keys():
      for graph_def in RegistPattern.get_patterns(pattern_name):
        delta_graph = DeltaGraph(pb_model=graph_def)
        RegisterPattern(pattern_name, delta_graph,
                        self.__hint_op_type(pattern_name))
    if pattern_name not in RegistPattern.Patterns().keys():
      raise ValueError(
          "Err: there isn't any pattern invoked within scope of GraphStream.")
    self.__optimizer = AutoOptimizer(self.__delta_graph_original)
    # run optimizer automatically
    self.__optimizer.run()

  def __hint_op_type(self, pattern_name):
    assert (pattern_name in self.__hint_map), \
            "Pattern name({}) with hint op must be registered by \
                function register_hint_op."                                                                                                                             .format(pattern_name)
    return self.__hint_map[pattern_name]

  def register_hint_op(self, pattern_name, hint_op_type):
    """ register hint op for pattern

        Arguments:
            pattern_name (string): name of this pattern
            hint_op_type (string): op type of of pattern,
                                   this param can be any op exist in pattern graph
        """
    self.__hint_map[pattern_name] = hint_op_type

  def save(self, path=None):
    """ set save path for final otimized graph by graph stream """
    self.__save_path = path if path is not None else "./result.pb"

  def __enter__(self):
    self.__delta_graph_original = DeltaGraph(pb_file=self.__path)
    return self

  def __exit__(self, type, value, traceback):
    self.__run()
    self.__optimizer.serialization(self.__save_path)
