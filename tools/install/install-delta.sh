#!/bin/bash

set -e

USAGE="USAGE: $0 [nlp|full] [gpu|cpu]"

if [ $# != 2 ];then
  echo ${USAGE}
  exit 1
fi

if [ $1 == 'nlp' ];then
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
  TF='tensorflow-gpu=1.14.0'
elif [ $2 == 'cpu' ];then
  TF='tensorflow=1.14.0'
else
  echo ${USAGE}
  exit 1
fi

# If you are a user from mainland China, you can use following codes to speed up the installation.
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

CONDA_ENV=delta-py3.6-tf1.14

if conda search ${TF} | grep "No match"; then
  echo "Conda: do not have ${TF} for current platform."
  if [ $2 == 'gpu' ];then
    TF='tensorflow-gpu=1.13.1'
  else
    TF='tensorflow=1.13.1'
  fi
  echo "Conda: install ${TF} instead."
  CONDA_ENV=delta-py3.6-tf1.13
fi

conda create -n ${CONDA_ENV} python=3.6
source activate ${CONDA_ENV}
echo "Conda: ${CONDA_ENV} activated!"

conda install ${TF}
echo "Conda: TensorFlow installed!"

CURRENT_VER="$(g++ -dumpversion)"
REQUIRE_VER="5.0.0"
if [ "$(printf '%s\n' "${REQUIRE_VER}" "${CURRENT_VER}" | sort -V | head -n1)" = "$REQUIRE_VER" ]; then
  echo "G++ version is ${CURRENT_VER}, which is greater than or equal to ${REQUIRE_VER}"
else
  printf "Tensorflow custom op compilation with G++ version less than 5.0.0 may be buggy.\\n"
  printf "Do you want to update the g++ version of this conda env? [yes|no]\\n"
  printf "[yes] >>> "
  read -r ans
  if [ "$ans" != "no" ] && [ "$ans" != "No" ] && [ "$ans" != "NO" ] && \
    [ "$ans" != "n" ] && [ "$ans" != "N" ]
    then
      conda install -c conda-forge cxx-compiler
  else
    echo "Skip g++ update."
  fi
fi

CONDA_PIP=`which pip`
if ! which ${CONDA_PIP}3; then
  ln -s ${CONDA_PIP} ${CONDA_PIP}3
fi

make ${TARGET}

echo "Required package installed! Compiling completed!"

echo "DELTA is successfully installed!"

echo "Please use \"conda activate ${CONDA_ENV}\" to activate the conda environment."
