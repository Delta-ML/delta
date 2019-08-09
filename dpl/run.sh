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

TARGET=$1
ARCH=$2

if [ -z $MAIN_ROOT ];then
  pushd ..
  source env.sh
  popd
  echo "source env.sh"
fi

INPUT_PATH="${MAIN_ROOT}/dpl/input_model"
OUTPUT_PATH="${MAIN_ROOT}/dpl/output_model"
MODEL_YAML="${MAIN_ROOT}/dpl/input_model/model.yaml"
VERSION=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.version`
ENGINE=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.engine`
MODEL_TYPE=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.local.model_type`
OUTPUT_NUM=`cat ${MODEL_YAML} | shyaml get-length model.graphs.0.outputs`

OUTPUT_NAMES=""
END_NUM=$((expr ${OUTPUT_NUM} - 1))

echo "OUTPUT_NUM: ${OUTPUT_NUM}"

for i in `seq 0 ${END_NUM}`
do
  NEW_OUTPUT=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.outputs.$i.name`
  OUTPUT_NAMES="${OUTPUT_NAMES},${NEW_OUTPUT}"
done

# 1. convert graph
# convert saved_model under `model` with `model.yaml`
# to `tf`, `tflite`, `tftrt` model 
# and save under `gadapter` dir

# 2. prepare third_party lib
# get and install lib under docker

# 3. compile tensorflow lib, tflite lib, tf-serving with custom op
# 4. compile deltann

#BAZEL_CACHE=../.cache/bazel
BAZEL_CACHE=${MAIN_ROOT}/tools/.cache/bazel
UTILS=${MAIN_ROOT}/dpl/utils/deploy

function convert_graph(){
  echo "Satrt transform graph ..."
  if [ ${ENGINE} == 'TF' ];then
    if [ ${MODEL_TYPE} == 'saved_model' ]; then
      GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/saved_model/${VERSION}"
      mkdir -p ${GADAPTER_PATH}
      cp -r ${INPUT_PATH}/* ${GADAPTER_PATH}
    elif [ ${MODEL_TYPE} == 'frozen_graph_pb' ]; then
      GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tfgraph"
      bash ${UTILS}/frozen_saved_model.sh ${GADAPTER_PATH} ${OUTPUT_NAMES}
    else
      echo "MODEL_TYPE: ${MODEL_TYPE} and ENGINE: ${ENGINE} error!"
      exit 1
    fi
  elif [ ${ENGINE} == 'TFLITE' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tflite"
    mkdir -p ${GADAPTER_PATH}
    echo "tflite to be added."
    exit 1
  elif [ ${ENGINE} == 'TFRT' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tfrt"
    mkdir -p ${GADAPTER_PATH}
    echo "tfrt to be added."
    exit 1
  elif [ ${ENGINE} == 'TFSERVING' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/saved_model/${VERSION}"
    mkdir -p ${GADAPTER_PATH}
    cp -r ${INPUT_PATH}/* ${GADAPTER_PATH}
  else
    echo "MODEL_TYPE: ${MODEL_TYPE} and ENGINE: ${ENGINE} error!"
    exit 1
  fi
  echo "Graph transformed."
}

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
  local arch=$2
  echo "Start compile tensorflow: $target $arch"

  if [ ${target} == 'linux' ] && [ ${arch} == 'x86_64' ];then
	pushd ${MAIN_ROOT}/tools/tensorflow
    bazel \
       build -c opt //tensorflow:libtensorflow_cc.so || exit 1
	
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
    bazel --output_user_root=$BAZEL_CACHE \
      build -c opt //tensorflow/lite/experimental/c:libtensorflowlite_c.so || exit 1

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
  local target=$2
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

# 1. convert graph 
# convert_graph

# 2. clear old libs
# clear_lib

# 3. compile tensorflow
# compile_tensorflow ${TARGET}  ${ARCH}

# 4. compile deltann
# compile_deltann ${TARGET} ${ARCH} ${ENGINE}

# 5. compile custom ops
compile_custom_ops tensorflow deltann

# 6. compile deltann egs
# compile_deltann_egs

# 7. run test
# run test under docker


