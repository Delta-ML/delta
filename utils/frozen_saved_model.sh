#!/bin/bash

. ../env.sh

if [[ $# != 2 ]];then
  echo "usage: $0 saved_model_dir output_nodes"
  echo "   Make sure delta/env.sh is sourced"
  exit 1
fi

# saved model dir
saved_model_dir=$1
output_nodes=$2

ROOT=${MAIN_ROOT}/tools/tensorflow
python3 ${ROOT}/tensorflow/python/tools/freeze_graph.py \
  --input_saved_model_dir=$saved_model_dir\
  --saved_model_tags='serve' \
  --output_node_names=$output_nodes\
  --output_graph='frozen_graph.pb'
