#!/bin/bash
set -ex

if [ $# != 2 ]; then
  echo "usage: $0 engine model_type version"
  echo "e.g. usage: $0 [ TF| TFLITE| TFTRT| TFSERVING] [saved_model | frozen_graph_pb] [version]"
  exit -1
fi

ENGINE=$1
MODEL_TYPE=$2
VERSION=$3

if [ -z $MAIN_ROOT ];then
  pushd ..
  source env.sh
  popd
  echo "source env.sh"
fi

function convert_graph(){
  engine=$1 # TF, TFLITE, TFTRT, TFSERVING
  model_type=$2
  version=$3
  echo "Satrt transform graph ..."

  if [ ${engine} == 'TF' ];then
    if [ ${model_type} == 'saved_model' ]; then
      GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/saved_model/${version}"
      cp -r ${INPUT_PATH}/*  ${GADAPTER_PATH}
    elif [ ${model_type} == 'frozen_graph_pb' ]; then
      GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tfgraph"
      bash ${UTILS}/frozen_saved_model.sh ${GADAPTER_PATH} ${OUTPUT_NAMES}
    else
      echo "MODEL_TYPE: ${model_type} and ENGINE: ${engine} error!"
      exit 1
    fi
  elif [ ${engine} == 'TFLITE' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tflite"
    echo "tflite to be added."
    exit 1
  elif [ ${engine} == 'TFTRT' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tfrt"
    echo "tfrt to be added."
    exit 1
  elif [ ${engine} == 'TFSERVING' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/saved_model/${version}"
    cp -r ${INPUT_PATH}/* ${GADAPTER_PATH}
  else
    echo "MODEL_TYPE: ${model_type} and ENGINE: ${engine} error!"
    exit 1
  fi
  echo "Graph transformed."
}

convert_graph $(ENGINE) $(MODEL_TYPE) $(VERSION)
