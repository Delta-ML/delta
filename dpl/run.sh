#!/bin/bash
set -e

# all work do under `docker`
# dpl input: 
#  from `model` dir which has `saved_model` dir 
#  and `model.yaml` config
# dpl output: 
#  graph: 
#    saved_model
#    model.tflite
#    model.tftrt
#    ...
#  TF Lib:
#    tensorflow so
#    tflite so
#    tf-searving bin
#  deltann:
#    deltann.so
#    deltann.out
# dpl test:
#  ci unittest 
#  offline model with one graph
#  offline model with multi graphs
#  online model
#  combined offline and online

if [ $# != 3 ]; then
    echo "usage: $0 [linux] [x86_64] [tf|tflite|tfserving]"
    exit -1
fi

TARGET=$1
ARCH=$2
ENGINE=$3

if [ -z $MAIN_ROOT ];then
    pushd ..
    source env.sh
    popd
    echo "source env.sh"
fi


# 1. convert graph
# convert saved_model under `model` with `model.yaml`
# to `tf`, `tflite`, `tftrt` model 
# and save under `gadapter` dir

# 2. prepare third_party lib
# get and install lib under docker

# 3. compile tensorflow lib, tflite lib, tf-serving with custom op
# 4. compile deltann

#BAZEL_CACHE=../.cache/bazel
BAZEL_CACHE=$MAIN_ROOT/tools/.cache/bazel

function clear_lib(){
  echo "clear library under dpl/lib "
  pushd $MAIN_ROOT/dpl/lib
  for dir in `ls`;
  do
      rm -rf $dir/* 
  done
  popd
}

function compile_tensorflow(){
  local target=$1 # linux
  local arch=$2
  echo "compile tensorflow: $target $arch"

  if [ $target == 'linux' ] && [ $arch == 'x86_64' ];then
	pushd $MAIN_ROOT/tools/tensorflow 
    bazel --output_user_root=$BAZEL_CACHE \
       build -c opt //tensorflow:libtensorflow_cc.so || exit 1
	
    pushd bazel-bin/tensorflow
    #if [ -L libtensorflow_cc.so.1 ]; then
    #    unlink libtensorflow_cc.so.1
    #fi
    #ln -s libtensorflow_cc.so.1 libtensorflow_cc.so
    #if [ -L libtensorflow_framework.so.1 ];then
    #    unlink libtensorflow_framework.so.1
    #fi
	#ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
	cp *.so* $MAIN_ROOT/dpl/lib/tensorflow/ 

    popd
    popd

  else
    echo "Not support: $target $arch"
    exit 1
  fi
}

function compile_tflite(){
  local target=$1
  local arch=$2
  echo "compile tflite: $target $arch"

  if [ $target == 'linux' ] && [ $arch == 'x86_64' ];then
     pushd $MAIN_ROOT/tools/tensorflow
     bazel --output_user_root=$BAZEL_CACHE \
       build -c opt //tensorflow/lite/experimental/c:libtensorflowlite_c.so || exit 1

	 cp tensorflow/bazel-bin/tensorflow/lite/experimental/c/*.so $MAIN_ROOT/dpl/lib/tflite/
     popd
  else
    echo "Not support: $target $arch"
    exit 1
  fi
}

function compile_custom_ops(){
  local platform=$1 # tensorflow
  local target=$2
  echo "compile custom ops: $platform $target"

  if [ $platform == 'tensorflow' ];then
      if [ $target != 'delta' ] && [ $target != 'deltann' ];then
          echo "compile custom error: target no support. "  $target
          exit 1
      fi

      pushd $MAIN_ROOT/delta/layers/ops/
      bash build.sh  $target
      popd
  else
      echo "Not support: $platform"
      exit 1
  fi
}

function compile_deltann(){
  local target=$1 # linux
  local arch=$2   # x86_64
  local engine=$3   # [tf|tflite|tfserving]
  
  pushd $MAIN_ROOT/deltann
  bash build.sh $target $arch $engine
  cp .gen/lib/* $MAIN_ROOT/dpl/lib/deltann
  popd
}

function compile_deltann_egs(){
    pushd $MAIN_ROOT/deltann
    make example
    popd
}

clear_lib

compile_tensorflow $TARGET  $ARCH
compile_deltann $TARGET $ARCH tf
compile_custom_ops tensorflow deltann


# compile_tflite $TARGET $ARCH
# compile_deltann $TARGET $ARCH tflite


compile_deltann_egs

# 5. run test
# run test under docker


