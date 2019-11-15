#!/bin/bash
# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


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

#prepare kaldi
if [ ! -d ${MAIN_ROOT}/tools/kaldi/tools/sph2pipe_v2.5 ]; then
  bash ${MAIN_ROOT}/tools/install/prepare_kaldi.sh
fi

echo "Integration Testing..."

pushd ${MAIN_ROOT}/egs/mini_an4/asr/v1
bash run_delta.sh || { echo "mini an4 error"; exit 1; }
popd

echo "Integration Testing Done."
