#!/bin/bash

CUDA_VISIBLE_DEVICES=1 ./bazel-bin/tensorflow/tools/benchmark/benchmark_model \
        --graph="/nfs/project/models/frozen_graph.pb" \
        --input_layer='inputs:0' \
        --input_layer_shape="32,1000,40,1" \
        --input_layer_type="float" \
        --input_layer_values="" \
        --output_layer='softmax_output:0'
