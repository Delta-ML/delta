#!/bin/bash

python3 convert_frozen_pb_to_tftrt.py \
    --input_graph ../model/emotion/frozen_graph_tf.pb \
    --out_tensor frozen_softmax  \
    --precision_mode FP32 \
    --batch_size 1024 \
    --gpu 2

