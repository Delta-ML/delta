#!/bin/bash

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

#USAGE="usage: $0 --TARGET=[linux] --ARCH=[x86_64]"

if [ -z $MAIN_ROOT ];then
  source ../env.sh
  echo "source env.sh"
fi

stage=-1
stop_stage=100

TARGET=linux
ARCH=x86_64 # or cuda

# input and output model dir
INPUT_MODEL="${MAIN_ROOT}/dpl/model"
INPUT_YAML="${INPUT_MODEL}/model.yaml"
OUTPUT_MODEL="${MAIN_ROOT}/dpl/.gen"

. ${MAIN_ROOT}/utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

echo
echo "Params:"
echo "stage: ${stage} - ${stop_stage}"
echo "TARGET: ${TARGET}"
echo "ARCH: ${ARCH}"
echo "INPUT_MODEL: ${INPUT_MODEL}"
echo "INPUT_YAML: ${INPUT_YAML}"
echo "OUTPUT_MODEL: ${OUTPUT_MODEL}"
echo

set -e
set -u
set -o pipefail

# 0. dpl and model config
# config from model.yaml
ENGINE=`cat ${INPUT_YAML} | shyaml get-value model.graphs.0.engine`

# 1. convert graph
# convert saved_model under `model` with `model.yaml`
# to `tf`, `tflite`, `tftrt` model
# and save under `gadapter` dir

# 2. prepare third_party lib
# get and install lib under docker

# 3. compile tensorflow lib, tflite lib, tf-serving with custom op
# 4. compile deltann

BAZEL_CACHE=${MAIN_ROOT}/tools/.cache/bazel
mkdir -p $BAZEL_CACHE
BAZEL="bazel --output_base=${BAZEL_CACHE}"
#BAZEL=bazel

function clear_lib(){
  echo "Clear library under dpl/lib..."
  pushd ${MAIN_ROOT}/dpl/lib
  for dir in `ls`;
  do
    rm -rf ${dir}/* && touch ${dir}/.gitkeep || exit 1
  done
  popd
  echo "Clear library done."
  echo
}

function compile_tensorflow(){
  local target=$1 # linux
  local arch=$2 #x86_64
  echo "Start compile tensorflow: $target $arch"

  OPTIONS="--verbose_failures -s -c opt --copt=-mavx --copt=-mfpmath=both --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2"
  pushd ${MAIN_ROOT}/tools/tensorflow

  if [ ${target} == 'linux' ] && [ ${arch} == 'x86_64' ];then
    ${BAZEL} build ${OPTIONS} //tensorflow:libtensorflow_cc.so || exit 1
    echo "Compile tensorflow cpu successfully."
  elif [ ${target} == 'linux' ] && [ ${arch} == 'gpu' ];then
    pushd ${MAIN_ROOT}/tools/tensorflow
    ${BAZEL} build ${OPTIONS} --config=cuda //tensorflow:libtensorflow_cc.so || exit 1
    echo "Compile tensorflow gpu successfully."
  else
    echo "Not support: $target $arch"
    exit 1
  fi

  pushd bazel-bin/tensorflow
  if [ ! -L libtensorflow_framework.so ];then
    ln -s libtensorflow_framework.so.2 libtensorflow_framework.so
  fi
  cp *.so* ${MAIN_ROOT}/dpl/lib/tensorflow/
  popd


  popd
  echo
}

function compile_tflite(){
  local target=$1
  local arch=$2
  echo "Start compile tflite: $target $arch"

  if [ ${target} == 'linux' ] && [ ${arch} == 'x86_64' ];then
    pushd ${MAIN_ROOT}/tools/tensorflow
    ${BAZEL} build -c opt --verbose_failures //tensorflow/lite/experimental/c:libtensorflowlite_c.so || exit 1

    cp tensorflow/bazel-bin/tensorflow/lite/experimental/c/*.so ${MAIN_ROOT}/dpl/lib/tflite/
    echo "Compile tensorflow lite successfully."
    popd
  else
    echo "Not support: $target $arch"
    exit 1
  fi
  echo
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
    bash build.sh ${target} || { echo "build ops error"; exit 1; }
    popd
    echo "Compile custom ops successfully."
  else
    echo "Not support: $platform"
    exit 1
  fi
  echo
}

function compile_deltann(){
  echo "Start compile deltann ..."
  local target=$1 # linux
  local arch=$2   # x86_64
  local engine=$3   # [tf|tflite|tfserving]

  pushd ${MAIN_ROOT}/deltann
  bash build.sh $target $arch $engine || { echo "build deltann error"; exit 1; }
  cp .gen/lib/* $MAIN_ROOT/dpl/lib/deltann || { echo "copy deltann error"; exit 1; }
  popd
  echo "Compile deltann successfully."
  echo
}

function compile_deltann_egs(){
  echo "Compile deltann examples..."
  pushd ${MAIN_ROOT}/deltann
  make examples || { echo "Compile deltann examples error"; exit 1; }
  popd
  echo "Compile deltann examples done."
  echo
}

function convert_model(){
  echo "Convert model..."
  pushd ${MAIN_ROOT}/dpl/gadapter
  bash run.sh || { echo "convert model error"; exit 1; }
  popd
  echo "Convert model done."
  echo
}

function dpl_output(){
  echo "dump output..."
  if [ -d output ]; then
      rm -rf output
  fi

  mkdir -p output/model/
  mkdir -p output/include/
  cp -R   lib/ output/

  cp -R  ../deltann/api/c_api.h  output/include/
  cp -R  gadapter/saved_model/ output/model/

  pushd output/model/saved_model/1/saved_model/
  mv saved_model.pb* ../
  mv variables ../
  cd ..
  rm -rf saved_model
  popd
  echo "dump output done."
  echo
}

function deltann_unit_test() {
  echo "deltann unit test ..."
  pushd $MAIN_ROOT/tools/test && ./cpp_test.sh && popd
  echo "deltann unit test done."
}

echo
echo "Input: ${INPUT_MODEL}"
echo "Output: ${OUTPUT_MODEL}"
echo

# 1. convert graph
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
  convert_model
fi

# 2. clear old libs
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
  clear_lib
fi

# 3. compile custom ops
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
  compile_custom_ops tensorflow deltann
fi

# 4. compile tensorflow
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ];then
  compile_tensorflow ${TARGET}  ${ARCH}
  # compile_tflite $TARGET $ARCH
fi

# 5. compile deltann
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ];then
  compile_deltann ${TARGET} ${ARCH} ${ENGINE}
  # compile_deltann $TARGET $ARCH tflite
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ];then
  deltann_unit_test
fi
 
# 6. compile deltann egs
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ];then
  compile_deltann_egs
fi

# 7. dump model and lib to `dpl/output`
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ];then
  dpl_output
fi

# 8. run test
# run test under docker

