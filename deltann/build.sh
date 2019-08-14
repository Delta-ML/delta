#!/bin/bash

if [ $# != 3 ];then
  echo "usage: $0 [linux|android|ios] [x86_64|arm|arm64] [tf|tflite|tfserving]"
  echo "now only support [linux] [x86_64] [TF|TFLITE|TFTRT|TFSERVING]"
  exit 1
fi

platform=$1
arch=$2
engine=$3

#TARGET=$platform TARGET_ARCH=$arch ENGINE=$engine make -f Makefile -e
export TARGET=$platform 
export TARGET_ARCH=$arch 
export ENGINE=$engine 

make clean
make -f Makefile -e
