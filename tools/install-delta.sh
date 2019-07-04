#!/bin/bash
CONDA_ENV=delta-py3.6-tf1.13
conda create -n ${CONDA_ENV} python=3.6
source activate ${CONDA_ENV}
echo "Conda: Python 3.6 activated!"

conda install tensorflow-gpu=1.13.1
echo "Conda: TensorFlow installed!"

CURDIR=`pwd`
cd tools && make basic
cd ${CURDIR}
echo "Required package installed! Compiling completed!"

source env.sh
echo "DELTA environment set up!"

echo "DELTA is successfully installed!" 
