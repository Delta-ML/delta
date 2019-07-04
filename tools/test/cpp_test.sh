#!/bin/bash

if [ -z $MAIN_ROOT ];then
  source env.sh
fi

pushd $MAIN_ROOT/tools/test && make clean && make && popd

$MAIN_ROOT/tools/test/test_main.bin
