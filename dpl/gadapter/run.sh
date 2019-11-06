#!/bin/bash

if [ $# != 0 ]; then
  echo "usage: $0"
  exit -1
fi

if [ -z $MAIN_ROOT ];then
  source ../../env.sh
  echo "source env.sh"
fi

# from `dpl/model/model.ymal`
# input and output model dir
INPUT_MODEL="${MAIN_ROOT}/dpl/model"
MODEL_YAML="${INPUT_MODEL}/model.yaml"

set -e
set -u
set -o pipefail

UTILS=${MAIN_ROOT}/dpl/utils/deploy

function convert_graph(){
  engine=$1 # TF, TFLITE, TFTRT, TFSERVING
  model_type=$2 # saved_model, frozen_graph_pb
  version=$3   # version
  input_model_path=$4 # input model path
  echo "Satrt transform graph ..."

  if [ ${engine} == 'TF' ];then
    if [ ${model_type} == 'saved_model' ]; then
      GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/saved_model/${version}"
      rm -rf $GADAPTER_PATH
      mkdir -p $GADAPTER_PATH
      echo "copy saved model to $GADAPTER_PATH"
      cp -r ${input_model_path}/*  ${GADAPTER_PATH} || { echo "copy saved_model error"; exit 1; }
    elif [ ${model_type} == 'frozen_graph_pb' ]; then
      echo "forzen graph"
      GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tfgraph"
      rm -rf $GADAPTER_PATH
      mkdir -p $GADAPTER_PATH
      bash ${UTILS}/frozen_saved_model.sh ${GADAPTER_PATH} ${OUTPUT_NAMES} || { echo "forzen graph error"; exit 1; }
    else
      echo "MODEL_TYPE: ${model_type} and ENGINE: ${engine} error!"
      exit 1
    fi
  elif [ ${engine} == 'TFLITE' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tflite"
    rm -rf $GADAPTER_PATH
    echo "tflite to be added."
    exit 1
  elif [ ${engine} == 'TFTRT' ];then
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/tfrt"
    rm -rf $GADAPTER_PATH
    echo "tfrt to be added."
    exit 1
  elif [ ${engine} == 'TFSERVING' ];then
    echo "copy saved model for tfserving"
    GADAPTER_PATH="${MAIN_ROOT}/dpl/gadapter/saved_model/${version}"
    rm -rf $GADAPTER_PATH
    cp -r ${input_model_path}/* ${GADAPTER_PATH} || { echo "copy saved model for tfserving error"; exit 1; }
  else
    echo "MODEL_TYPE: ${model_type} and ENGINE: ${engine} error!"
    exit 1
  fi
  echo "Graph transformed."
}

# 0. dpl and model config 
# config from model.yaml
VERSION=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.version`
ENGINE=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.engine`
MODEL_TYPE=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.local.model_type`
OUTPUT_NUM=`cat ${MODEL_YAML} | shyaml get-length model.graphs.0.outputs`

echo "OUTPUT_NUM: ${OUTPUT_NUM}"

OUTPUT_NAMES=""
END_NUM=$((${OUTPUT_NUM} - 1 ))

for i in `seq 0 ${END_NUM}`
do
  NEW_OUTPUT=`cat ${MODEL_YAML} | shyaml get-value model.graphs.0.outputs.$i.name`
  OUTPUT_NAMES="${OUTPUT_NAMES},${NEW_OUTPUT}"
done

echo "OUTPUT_NAMES: ${OUTPUT_NAMES}"

convert_graph ${ENGINE} $MODEL_TYPE $VERSION ${INPUT_MODEL}
