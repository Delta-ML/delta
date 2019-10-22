#! /bin/bash

export DELTANN_MAIN=../dpl/output/lib/deltann
export DELTANN_OPS=../dpl/output/lib/custom_ops
export DELTANN_TENSORFLOW=../dpl/output/lib/tensorflow

LD_LIBRARY_PATH=$DELTANN_MAIN:$DELTANN_OPS:$DELTANN_TENSORFLOW:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

