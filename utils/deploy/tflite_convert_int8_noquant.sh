#!/bin/bash

set -x

if [[ $# != 1 ]]; then
  echo "usage: $0 saved_model_path"
  exit 1
fi

mkdir -p graph_dump_int8_noquant

CUDA_VISIBLE_DEVICES= tflite_convert --output_file=graph_int8_noquant.tflite \
  --saved_model_dir=$1 \
  --output_format TFLITE \
  --inference_type QUANTIZED_UINT8 \
  --inference_input_type QUANTIZED_UINT8 \
  --input_shapes=-1, 3000, 40, 3 \
  --mean_values='127.0' \
  --std_dev_values='0.0078125' \
  --default_ranges_max=6 \
  --default_ranges_min=0 \
  --dump_graphviz_dir graph_dump_int8_noquant
