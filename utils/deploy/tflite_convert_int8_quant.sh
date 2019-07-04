#!/bin/bash

if [[ $# != 1 ]]; then
  echo "usage: $0 saved_model_path"
  exit 1
fi

tflite_convert --output_file=graph_int8_quant.tflite \
  --saved_model_dir=$1 \
  --output_format TFLITE \
  --inference_type QUANTIZED_UINT8 \
  --inference_input_type QUANTIZED_UINT8 \
  --std_dev_values=1.0 \
  --mean_values=0 \
  --default_ranges_max=6 \
  --default_ranges_min=0 \
  --post_training_quantize \
  --dump_graphviz_dir graph_dump_int8_quant
