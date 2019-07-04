#!/bin/bash
# saved model dir
set -x 

if [[ $# != 1 ]]; then
  echo "usage: $0 saved_model_dir"
  exit 1
fi

saved_model_dir=$1

ROOT=${MAIN_ROOT}/tools/tensorflow
${ROOT}/bazel-bin/tensorflow/python/tools/saved_model_cli show --dir=$saved_model_dir --all
