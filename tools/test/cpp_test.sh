#!/bin/bash

if [ -z $MAIN_ROOT ];then
  source env.sh
fi

pushd $MAIN_ROOT/tools/test && make clean && make && popd

LD_LIBRARY_PATH=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') $MAIN_ROOT/tools/test/test_main.bin
