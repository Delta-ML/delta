#!/bin/bash

make examples

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../dpl/lib/tensorflow:$PWD/../dpl/lib/deltann/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../dpl/lib/tensorflow:$PWD/.gen/lib

echo "text_cls"
./examples/text_cls/test.bin examples/text_cls/model.yaml

echo "speaker"
./examples/speaker/test.bin examples/speaker/model.yaml
