#!/bin/bash

graph=$1

ROOT=${MAIN_ROOT}/tools/tensorflow

${ROOT}/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$graph --print_structure=false
