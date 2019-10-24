#!/bin/bash

make examples

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../dpl/lib/tensorflow:$PWD/../dpl/lib/deltann/
./examples/text_cls/test.bin examples/text_cls/model.yaml
