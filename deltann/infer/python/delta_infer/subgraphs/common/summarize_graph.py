from __future__ import absolute_import, division, print_function
import tensorflow as tf

import os
import sys
import numpy as np
from tensorflow.compat.v1 import GraphDef

__all__ = ["GraphSummary"]

class GraphSummary(object):
    """ Graph Def summary """
    def __init__(self, graph_pb_path=None, graph_def=None):
        if graph_pb_path is not None:
            with tf.compat.v1.gfile.GFile(graph_pb_path, 'rb') as f:
                self.graph = tf.compat.v1.GraphDef()
                self.graph.ParseFromString(f.read())
        else:
            self.graph = graph_def
        self.summray_dict = {}

    @property
    def graph(self):
        return self.graph_def

    @graph.setter
    def graph(self, graph):
        if graph is not None:
            if isinstance(graph, GraphDef):
                self.graph_def = graph
            else:
                raise ValueError("graph({}) should be type of GraphDef.".format(type(graph)))


    def PrintNodeInfo(self, node):
        get_real_shape = lambda dims: [ dim.size for dim in dims]
        if "shape" in node.attr:
            shape = get_real_shape(node.attr["shape"].shape.dim)
        if "dtype" in node.attr:
            dtype = tf.DType(node.attr["dtype"].type)
        print("    <<=== (name={}, type={}, shape={} )".format(node.name, dtype, shape))
        return (node.name, dtype, shape)

    def MapNodesToOutputs(self, output_map):
        for node in self.graph.node:
            for input in node.input:
                output_map[input.split(":")[0]] = node

    def __getitem__(self, key):
        return self.summray_dict[key]

    def Summary(self):
        placeholders = []
        variables = []
        print("Graph Version: {}.{}".format(self.graph.versions.producer, self.graph.versions.min_consumer))
        for node in self.graph.node:
            if node.op == "Placeholder":
                placeholders.append(node)
            if node.op == "Variable" or node.op == "VariableV2":
                variables.append(node)
        if len(placeholders) == 0:
            print("No inputs spotted")
        else:
            print("Found {} possible inputs: ".format(len(placeholders)))
            self.summray_dict["inputs"] = []
            for node in placeholders:
                in_info = self.PrintNodeInfo(node)
                self.summray_dict["inputs"].append(in_info)
        if len(variables) == 0:
            pass
            print("No variables spotted")
        else:
            print("Found {} variables".format(len(variables)))
            self.summray_dict["variables"] = []
            for node in variables:
                var_info = self.PrintNodeInfo(node)
                self.summray_dict["variables"].append(var_info)

        output_map = {}
        self.MapNodesToOutputs(output_map)
        outputs = []
        unlikely_output_types = ["Const", "Assign", "NoOp", "Placeholder", "VarIsInitializedOp"]
        for node in self.graph.node:
            if (node.name not in output_map) and (node.op not in unlikely_output_types):
                outputs.append(node)
        if len(outputs) == 0:
            print("No outputs spotted")
        else:
            print("Found {} possible outputs:".format(len(outputs)))
            self.summray_dict["outputs"] = []
            for node in outputs:
                print("    ===>> (name={}, op={})".format(node.name, node.op))
                self.summray_dict["outputs"].append(node.name)

        const_parameter_count = 0
        variable_parameter_count = 0
        control_edge_count = 0
        device_counts = {}
        for node in self.graph.node:
            for input in node.input:
                if input[0] == "^":
                    control_edge_count+=1
            if len(node.device)!=0:
                device_counts[node.device] = 0 if node.device not in device_counts else device_counts[node.device]+1
            if node.op in ["Const", "Variable", "VariableV2"]:
                if "value" in node.attr:
                    tensor = tf.io.parse_tensor(node.attr["value"].tensor.SerializeToString(), tf.DType(node.attr["value"].tensor.dtype))
                    num_elements = tensor.shape.num_elements()
                    if node.op == "Const":
                        const_parameter_count += num_elements if num_elements is not None else 0
                    else:
                        variable_parameter_count += num_elements
        self.summray_dict["const_parameter_count"] = const_parameter_count
        self.summray_dict["variable_parameter_count"] = variable_parameter_count
        self.summray_dict["control_edge_count"] = control_edge_count
        print("Found {} const parameters, {} variable parameters, and {} control_edges".format(
            const_parameter_count, variable_parameter_count, control_edge_count))
        if len(device_counts.keys()) != 0:
            str_dev_info = ""
            for device_info in device_counts:
                str_dev_info += "%s nodes assigned to device %s, " % (str(device_info.second), str(device_info.first))
            print(str_dev_info)

        #op_counts = {}
        #for node in self.graph.node:
        #    op_counts[node.op] = 1 if node.op not in op_counts else op_counts[node.op]+1
        #for function in self.graph.library.function:
        #    for node in function.node_def.node:
        #        op_counts[node.op] = 1 if node.op not in op_counts else op_counts[node.op]+1
        #print("Op types used: ")
        #self.summray_dict["OpCount"] = {}
        #for op in op_counts:
        #    print("@ {} : {}".format(op, op_counts[op]))
        #self.summray_dict["OpCount"] = op_counts[op]
