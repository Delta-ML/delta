#!/usr/bin/env bash

PIP_NAME="delta-nlp"

echo "Uninstall ${PIP_NAME} if exist ..."
pip3 uninstall -y ${PIP_NAME}

echo "Build binary distribution wheel file ..."
BASH_DIR=`dirname "$BASH_SOURCE"`
pushd ${BASH_DIR}/../..
rm -rf build/ ${PIP_NAME}.egg-info/ dist/
python3 setup.py bdist_wheel
popd
