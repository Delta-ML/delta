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
  TARGET='all'
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

# https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

CONDA_ENV=delta-py3.6-tf1.14
conda create -n ${CONDA_ENV} python=3.6
source activate ${CONDA_ENV}
echo "Conda: Python 3.6 activated!"

conda install ${TF}
echo "Conda: TensorFlow installed!"

CONDA_PIP=`which pip | xargs realpath`
ln -s ${CONDA_PIP} ${CONDA_PIP}3

make ${TARGET}

echo "Required package installed! Compiling completed!"

echo "DELTA is successfully installed!" 
