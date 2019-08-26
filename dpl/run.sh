#!/bin/bash
set -ex

# Please run this script using `docker`
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

USAGE="usage: $0 [linux] [x86_64]"

if [ $# != 2 ]; then
  echo ${USAGE}
  exit -1
fi

if [ -z $MAIN_ROOT ];then
  source ../env.sh
  echo "source env.sh"
fi

TARGET=$1
ARCH=$2

# input and output model dir
INPUT_MODEL="${MAIN_ROOT}/dpl/model"
MODEL_YAML="${INPUT_MODEL}/model.yaml"
OUTPUT_MODEL="${MAIN_ROOT}/dpl/.gen"

. ${MAIN_ROOT}/utils/parse_options.sh  # e.g. this parses the --stage option if supplied.


# 0. dpl and model config 
# config from model.yaml
ENGINE=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.engine`

# 1. convert graph
# convert saved_model under `model` with `model.yaml`
# to `tf`, `tflite`, `tftrt` model 
# and save under `gadapter` dir

# 2. prepare third_party lib
# get and install lib under docker

# 3. compile tensorflow lib, tflite lib, tf-serving with custom op
# 4. compile deltann

BAZEL_CACHE=${MAIN_ROOT}/tools/.cache/bazel
#BAZEL=bazel --output_user_root=$BAZEL_CACHE
BAZEL=bazel
UTILS=${MAIN_ROOT}/dpl/utils/deploy

function clear_lib(){
  echo "Clear library under dpl/lib"
  pushd ${MAIN_ROOT}/dpl/lib
  for dir in `ls`;
  do
    rm -rf ${dir}/* && touch ${dir}/.gitkeep
  done
  echo "Clear library done."
  popd
}

function compile_tensorflow(){
  local target=$1 # linux
  local arch=$2 #x86_64
  echo "Start compile tensorflow: $target $arch"

  if [ ${target} == 'linux' ] && [ ${arch} == 'x86_64' ];then
	pushd ${MAIN_ROOT}/tools/tensorflow
    $(BAZEL) build -c opt //tensorflow:libtensorflow_cc.so || exit 1
	
    pushd bazel-bin/tensorflow
    #if [ -L libtensorflow_cc.so.1 ]; then
    #    unlink libtensorflow_cc.so.1
    #fi
    #ln -s libtensorflow_cc.so.1 libtensorflow_cc.so
    if [ -L libtensorflow_framework.so.1 ];then
      unlink libtensorflow_framework.so.1
    fi
    ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
    # cp *.so* ${MAIN_ROOT}/dpl/lib/tensorflow/
    echo "Compile tensorflow successfully."
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
  echo "Start compile tflite: $target $arch"

  if [ ${target} == 'linux' ] && [ ${arch} == 'x86_64' ];then
    pushd ${MAIN_ROOT}/tools/tensorflow
    $(BAZEL) build -c opt //tensorflow/lite/experimental/c:libtensorflowlite_c.so || exit 1

    cp tensorflow/bazel-bin/tensorflow/lite/experimental/c/*.so ${MAIN_ROOT}/dpl/lib/tflite/
    echo "Compile tensorflow lite successfully."
    popd
  else
    echo "Not support: $target $arch"
    exit 1
  fi
}

function compile_custom_ops(){
  local platform=$1 # tensorflow
  local target=$2 #delta, deltann
  echo "Strat compile custom ops: $platform $target"

  if [ ${platform} == 'tensorflow' ];then
    if [ ${target} != 'delta' ] && [ ${target} != 'deltann' ];then
      echo "compile custom error: target no support. "  ${target}
          exit 1
    fi

    pushd ${MAIN_ROOT}/delta/layers/ops/
    bash build.sh ${target}
    popd
    echo "Compile custom ops successfully."
  else
    echo "Not support: $platform"
    exit 1
  fi
}

function compile_deltann(){
  echo "Start compile deltann ..."
  local target=$1 # linux
  local arch=$2   # x86_64
  local engine=$3   # [tf|tflite|tfserving]
  
  pushd ${MAIN_ROOT}/deltann
  bash build.sh $target $arch $engine
  cp .gen/lib/* $MAIN_ROOT/dpl/lib/deltann
  popd
  echo "Compile deltann successfully."
}

function compile_deltann_egs(){
  pushd ${MAIN_ROOT}/deltann
  make example
  popd
}

sudo chown -R deltann:deltann $MAIN_ROOT/tools
sudo chown -R deltann:deltann $MAIN_ROOT/dpl

echo "Input: ${INPUT_MODEL}"
echo "Output: ${OUTPUT_MODEL}"

# 1. convert graph 
pushd ${MAIN_ROOT}/dpl/gadapter && bash run.sh && popd

# 2. clear old libs
clear_lib

# 3. compile tensorflow
compile_tensorflow ${TARGET}  ${ARCH}
# compile_tflite $TARGET $ARCH

# 4. compile deltann
compile_deltann ${TARGET} ${ARCH} ${ENGINE}
# compile_deltann $TARGET $ARCH tflite

# 5. compile custom ops
compile_custom_ops tensorflow deltann

# 6. compile deltann egs
compile_deltann_egs

# 7. dump model and lib to `output_model`

# 8. run test
# run test under docker

