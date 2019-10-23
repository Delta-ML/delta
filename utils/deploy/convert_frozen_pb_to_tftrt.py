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

#frozen graph tf version must be same to this tf version
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import delta.compat as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

workspace_size = 1 << 30


def getFrozenGraph(input_graph):
  with gfile.FastGFile(input_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def getFP32(input_graph, out_tensor, precision, batch_size, workspace_size):
  graph_prefix = input_graph.split('.pb')[0]
  output_graph = graph_prefix + "_tftrt_" + precision + ".pb"
  #print("output graph is ", output_graph)
  tftrt_graph = trt.create_inference_graph(
      getFrozenGraph(input_graph), [out_tensor],
      max_batch_size=batch_size,
      max_workspace_size_bytes=workspace_size,
      precision_mode=precision)  # Get optimized graph
  with gfile.FastGFile(output_graph, 'wb') as f:
    f.write(tftrt_graph.SerializeToString())


if "__main__" in __name__:
  P = argparse.ArgumentParser(description="tftrt grpah convert tool!!")
  P.add_argument('--input_graph', help="input tf frozen_gaph.pb")
  P.add_argument('--out_tensor', help="output tensor name ")
  P.add_argument('--precision_mode', help="FP32, FP16, INT8")
  P.add_argument('--batch_size', type=int, default=1024)
  P.add_argument(
      '--workspace_size',
      type=int,
      default=1 << 30,
      help="workspace size in MB")
  P.add_argument('--gpu', default=0, help="select gpu")

  f = P.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = f.gpu

  input_graph = f.input_graph
  out_tensor = f.out_tensor
  precision = f.precision_mode
  batch_size = f.batch_size
  workspace_size = f.workspace_size

  print("input graph is ", input_graph)
  print("output tensor is ", out_tensor)
  print("output precision_mode is ", precision)
  print("batch_size is ", batch_size)
  print("workspace_size is ", workspace_size)

  getFP32(input_graph, out_tensor, precision, batch_size, workspace_size)
