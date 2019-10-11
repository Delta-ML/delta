#!/bin/bash

BASH_DIR=`dirname "$BASH_SOURCE"`

pushd ${BASH_DIR}/../..
source env.sh
popd

pushd ${MAIN_ROOT}/tools
make deltann || echo "deltann install failed" && exit -1
popd

echo "deltann has been installed successfully ~"
