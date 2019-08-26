#!/bin/bash

if [ -z $MAIN_ROOT ];then
  if [ -f env.sh ];then 
      source env.sh
  else
      source ../../env.sh
  fi
fi


echo "Integration Testing..."

pushd $MAIN_ROOT/egs/mini_an4/asr/v1 && bash run_delta.sh && popd

echo "Integration Testing Done."
