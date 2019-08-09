#!/bin/bash

BASH_DIR=`dirname "$BASH_SOURCE"`

pushd ${BASH_DIR}/../..
source env.sh
popd

make deltann

echo "deltann has been installed successfully ~"
