from __future__ import absolute_import, division, print_function
import export_py as ep
import tensorflow as tf

__all__ = ["DeltaGraph", "AutoOptimizer", "RegisterPattern"]


class LocalOptimizerMgr(object):
  """ Note:
        not work on global static initialization used in c++ lib.
    """

  def __init__(self):
    pass

  @staticmethod
  def names():
    mgr = ep.optimizer.OptimizerManager.Instance()
    return mgr.names()

  @staticmethod
  def method(name):
    mgr = ep.optimizer.OptimizerManager.Instance()
    if mgr.contain(name):
      return mgr.Get(name)
    else:
      return None


def RegisterPattern(pattern_name, pattern, hint_op_type):
  """ """
  assert isinstance(pattern, DeltaGraph), \
          "type of pattern must be instance of DeltaGraph. but got {}".format(type(pattern))
  ep.optimizer.RegisterFusionPattern(pattern_name, pattern.pattern(),
                                     hint_op_type)


class DeltaGraph(object):
  """ """

  def __init__(self, pb_file=None, pb_model=None):
    """ """
    self.__pattern = ep.core.Pattern()
    if pb_file is not None:
      self.__pattern.LoadModel(pb_file)
    elif pb_model is not None:
      print(type(pb_model))
      assert isinstance(pb_model, tf.compat.v1.GraphDef), \
              "type of pb_model must be instance of GraphDef, but got {}".format(type(pb_model))
      self.__pattern.LoadModelCT(pb_model.SerializeToString())
    else:
      raise ValueError("Err: pb_file and pb_model can't be both None! ")

  def pattern(self):
    return self.__pattern

  @property
  def graph(self):
    return self.__pattern.graph()


class AutoOptimizer(object):
  """ """

  def __init__(self, graph):
    """ """
    assert isinstance(graph, DeltaGraph), \
            " type of Parameter graph must be instance of DeltaGraph. but got {}".format(type(graph))
    self.__optimizer = ep.optimizer.LocalAutoOptimizer(graph.pattern())

  def run(self):
    """ run opitimization automatically """
    self.__optimizer.run()

  def serialization(self, path):
    """ serialization to path """
    self.__optimizer.serialization(path)
