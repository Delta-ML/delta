#!/bin/bash
if [ -z ${MAIN_ROOT} ];then
  if [ -f env.sh ];then
      source env.sh
  else
      source ../../env.sh
  fi
fi

pushd ${MAIN_ROOT}/tools/kaldi/tools
#sudo apt-get install zlib1g-dev wget subversion
extras/check_dependencies.sh

SPH2PIPE_VERSION=v2.5
wget -T 10 -t 3 https://www.openslr.org/resources/3/sph2pipe_${SPH2PIPE_VERSION}.tar.gz || wget -T 10 https://sourceforge.net/projects/kaldi/files/sph2pipe_${SPH2PIPE_VERSION}.tar.gz || exit 1
tar --no-same-owner -xzf sph2pipe_v2.5.tar.gz
cd sph2pipe_v2.5/
gcc -o sph2pipe  *.c -lm
