#!/bin/bash

if [ -z ${MAIN_ROOT} ];then
  if [ -f env.sh ];then 
      source env.sh
  else
      source ../../env.sh
  fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

echo "Integration Testing..."

pushd ${MAIN_ROOT}/egs/mini_an4/asr/v1
bash run_delta.sh || echo "mini an4 error" && exit 1
popd

echo "Integration Testing Done."
