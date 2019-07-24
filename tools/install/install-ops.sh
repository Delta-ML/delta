#!/bin/bash

if [ -z $MAIN_ROOT ];then
 source ../../../env.sh
fi

rm $MAIN_ROOT/tools/compile_ops.done || true

pushd $MAIN_ROOT/tools && make compile_ops.done && popd
