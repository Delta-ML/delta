#!/bin/bash

if [ $# != 1 ];then
    echo "usage: $0 [delta|deltann]"
    exit 1
fi

target=$1

if [ -z $MAIN_ROOT ];then
    pushd ../../../ && source env.sh && popd
    echo "source env.sh"
fi

set -e
set -u
set -o pipefail

# prepare dependency
echo "build ops: prepare dependency"

if [ -L $MAIN_ROOT/delta/layers/ops/cppjieba ]; then
    unlink $MAIN_ROOT/delta/layers/ops/cppjieba
fi

if ! [ -f $MAIN_ROOT/tools/cppjieba.done ]; then
  pushd $MAIN_ROOT/tools && make cppjieba.done && popd
fi

ln -s $MAIN_ROOT/tools/cppjieba $MAIN_ROOT/delta/layers/ops/cppjieba || { echo "build ops: link jieba error" ; exit 1; }

# clean 

make clean &> /dev/null

# compile custom ops under tensorflow/core/user_ops

if [ $target == 'delta' ];then
    make -j $(nproc)

    if [ ! -f ./x_ops.so ];then
        echo "No x_ops.so generated. Compiling ops failed!"
        exit 1
    fi

elif [ $target == 'deltann' ]; then
    if [ -L $MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops/ops ];then
        unlink $MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops/ops
    fi
    ln -s $MAIN_ROOT/delta/layers/ops  $MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops 
    
    pushd $MAIN_ROOT/tools/tensorflow
   
    bazel build --verbose_failures -c opt //tensorflow/core/user_ops/ops:x_ops.so || { echo "compile custom ops error"; exit 1; }
    
    cp bazel-bin/tensorflow/core/user_ops/ops/*.so $MAIN_ROOT/dpl/lib/custom_ops
    cp $MAIN_ROOT/dpl/lib/custom_ops/x_ops.so $MAIN_ROOT/delta/layers/ops/ 

    popd
fi
