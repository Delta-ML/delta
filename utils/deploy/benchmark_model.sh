#!/bin/bash

ROOT=${MAIN_ROOT}/tools/tensorflow

${ROOT}/bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=frozen_graph.pb --show_flops --input_layer=inputs --input_layer_type=float --input_layer_shape=10,3000,40,3 --output_layer=softmax_output
