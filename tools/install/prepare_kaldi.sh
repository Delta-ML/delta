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

if [ -z ${MAIN_ROOT} ];then
  if [ -f env.sh ];then
      source env.sh
  else
      source ../../env.sh
  fi
fi

if [ `id -u` == 0 ];then
  SUDO=sudo
else
  SUDO=
fi

if ! [ -d ${MAIN_ROOT}/tools/kaldi ];then
  pushd ${MAIN_ROOT}/tools && git clone --depth=1 https://github.com/kaldi-asr/kaldi.git && popd
fi


pushd ${MAIN_ROOT}/tools/kaldi/tools
#sudo apt-get install zlib1g-dev wget subversion
extras/check_dependencies.sh || ${SUDO} apt-get install -y zlib1g-dev wget gfortran subversion

SPH2PIPE_VERSION=v2.5
test -e sph2pipe_${SPH2PIPE_VERSION}.tar.gz && rm sph2pipe_${SPH2PIPE_VERSION}.tar.gz
wget -T 10 -t 3 https://www.openslr.org/resources/3/sph2pipe_${SPH2PIPE_VERSION}.tar.gz || wget -T 10 https://sourceforge.net/projects/kaldi/files/sph2pipe_${SPH2PIPE_VERSION}.tar.gz || exit 1
tar --no-same-owner -xzf sph2pipe_v2.5.tar.gz
cd sph2pipe_${SPH2PIPE_VERSION}; patch -p1 < ${MAIN_ROOT}/tools/install/sph2pipe.patch; gcc -o sph2pipe  *.c -lm; cd -
touch  sph2pipe_${SPH2PIPE_VERSION}/.patched
popd
