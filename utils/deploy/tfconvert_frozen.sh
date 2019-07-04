#!/bin/bash
# convert frozen graph to tflite model

tflite_convert \
    --output_file=foo.tflite \
    --graph_def_file=frozen_graph_tflite.pb \
    --input_arrays=feat \
    --output_arrays=softmax_output
