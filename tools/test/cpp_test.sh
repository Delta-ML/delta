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

if [ -z $MAIN_ROOT ];then
  if [ $curdir == "." ];then
    source ../../env.sh
  else
    source env.sh
  fi
fi

if [ ! -d $MAIN_ROOT/deltann/.gen ];then
  pushd $MAIN_ROOT/deltann && ./build.sh linux x86_64 TF && popd
fi

pushd $MAIN_ROOT/tools/test && make clean && make && popd

set -x

#echo $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#LD_LIBRARY_PATH=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') $MAIN_ROOT/tools/test/test_main.bin

LD_LIBRARY_PATH=$MAIN_ROOT/tools/tensorflow/bazel-bin/tensorflow  $MAIN_ROOT/tools/test/test_main.bin
