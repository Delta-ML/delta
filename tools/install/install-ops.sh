#!/bin/bash

BASH_DIR=`dirname "$BASH_SOURCE"`

if [ -z $MAIN_ROOT ];then
  pushd ${BASH_DIR}/../../
  source env.sh
  popd
fi

rm $MAIN_ROOT/tools/compile_ops.done || true

pushd $MAIN_ROOT/tools && make compile_ops.done && popd
