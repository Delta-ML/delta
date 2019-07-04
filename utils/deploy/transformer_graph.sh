#!/bin/bash
graph=$1

# strip_unused_nodes shape for input placeholder 
ROOT=${MAIN_ROOT}/tools/tensorflow

${ROOT}/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=$graph \
--out_graph=transformed_graph.pb \
--inputs='input:0' \
--outputs='softmax_output:0' \
--transforms='
  add_default_attributes
  strip_unused_nodes(type=float, shape="-1,480000")
  remove_nodes(op=CheckNumerics)
  fold_batch_norms
  fold_old_batch_norms
  quantize_weights
  quantize_nodes
  strip_unused_nodes
  sort_by_execution_order
  '
  #fold_constants(clear_output_shapes=false, ignore_errors=true) # error
  #obfuscate_names # ok
