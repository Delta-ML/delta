#!/bin/bash

set -e

USAGE="USAGE: $0 [nlp|basic|full] [gpu|cpu]"

if [ $# != 2 ];then
  echo ${USAGE}
  exit 1
fi

# TF_VER, PY_VER
BASH_DIR=`dirname "$BASH_SOURCE"`

pushd ${BASH_DIR}/../..
source env.sh
popd

if [ $1 == 'nlp' ];then
  TARGET='nlp'
elif [ $1 == 'basic' ];then
  TARGET='basic'
elif [ $1 == 'full' ];then
  if [ -n ${KALDI} ]; then
    TARGET="delta KALDI=${KALDI}"
  else
    TARGET='delta'
  fi
else
  echo ${USAGE}
  exit 1
fi

if [ $2 == 'gpu' ];then
  TF="tensorflow-gpu==${TF_VER}"
elif [ $2 == 'cpu' ];then
  TF="tensorflow==${TF_VER}"
else
  echo ${USAGE}
  exit 1
fi

# If you are a user from mainland China, you can use following codes to speed up the installation.
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

CONDA_ENV=delta-py${PY_VER}-tf${TF_VER}

conda create -n ${CONDA_ENV} python=${PY_VER}
source activate ${CONDA_ENV}
echo "Conda: ${CONDA_ENV} activated!"


if [ $2 == 'gpu' ];then
  conda install cudnn=7.6.0
  conda install cudatoolkit=10.0.130
fi

echo "pip install ${TF}"
pip install ${TF}
echo "Conda: TensorFlow installed!"

python -c 'import tensorflow as tf; print(f"Tensorflow compiler version: {tf.version.COMPILER_VERSION}")'

# GCC Update should be turned off now
#CURRENT_VER="$(g++ -dumpversion)"
#REQUIRE_VER="5.0.0"
#if [ "$(printf '%s\n' "${REQUIRE_VER}" "${CURRENT_VER}" | sort -V | head -n1)" = "$REQUIRE_VER" ]; then
#  echo "G++ version is ${CURRENT_VER}, which is greater than or equal to ${REQUIRE_VER}"
#else
#  printf "Tensorflow custom op compilation with G++ version less than 5.0.0 may be buggy.\\n"
#  printf "Do you want to update the g++ version of this conda env? [yes|no]\\n"
#  printf "[yes] >>> "
#  read -r ans
#  if [ "$ans" != "no" ] && [ "$ans" != "No" ] && [ "$ans" != "NO" ] && \
#    [ "$ans" != "n" ] && [ "$ans" != "N" ]
#    then
#      conda install -c conda-forge cxx-compiler
#  else
#    echo "Skip g++ update."
#  fi
#fi

CONDA_PIP=`which pip`
if ! which ${CONDA_PIP}3; then
  ln -s ${CONDA_PIP} ${CONDA_PIP}3
fi

pushd ${BASH_DIR}/../../tools
make ${TARGET}
popd

echo "Required package installed! Compiling completed!"

echo "DELTA is successfully installed!"

echo "Please use \"conda activate ${CONDA_ENV}\" to activate the conda environment."
