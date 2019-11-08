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

set -e

curdir="$(dirname "${BASH_SOURCE[0]}")"

mode=docker
if [ ! -z $1 ];then
mode=$1
fi

echo "build mode $mode"

if [ $mode == "delta" ];then
  if [ -z $MAIN_ROOT ];then
    if [ $curdir == "." ];then
      source ../../env.sh
    else
      source env.sh
    fi
  fi
  BAZEL_CACHE=${MAIN_ROOT}/tools/.cache/bazel
  mkdir -p $BAZEL_CACHE
  BAZEL="bazel --output_base=${BAZEL_CACHE}"
  pushd $MAIN_ROOT/tools/tensorflow
  cp -r $MAIN_ROOT/.pip ~
else
  BAZEL="bazel"
  pushd /tensorflow_src
fi

echo "build_pip_package..."
$BAZEL build //tensorflow/tools/pip_package:build_pip_package

echo "build tensorflow package..."
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg 

echo "install tensorflow package..."
pip install ../tensorflow_pkg/tensorflow*.whl

if ! [ $mode == 'delta' ];then
echo "bazel clean..."
$BAZEL clean
fi

popd
