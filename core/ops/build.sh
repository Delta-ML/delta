#!/bin/bash

if [ $# != 1 ];then
    echo "usage: $0 [delta|deltann]"
    exit 1
fi

target=$1

if [ -z $MAIN_ROOT ];then
    pushd ../../ && source env.sh && popd
    echo "source env.sh"
fi

set -e
set -u
set -o pipefail

# check tf compiler version
local_ver=`gcc --version | grep ^gcc | sed 's/^.* //g'`
tf_ver=`python -c "import tensorflow as tf; print(tf.version.COMPILER_VERSION.split()[0]);"`

if [  ${local_ver:0:1} -ne ${tf_ver:0:1} ];then
  echo "gcc version($local_ver) not compatiable with tf compile version($tf_ver)"
  exit -1
fi


# prepare dependency
echo "build ops: prepare dependency"

if [ -L $MAIN_ROOT/core/ops/cppjieba ]; then
    unlink $MAIN_ROOT/core/ops/cppjieba
fi

if ! [ -f $MAIN_ROOT/tools/cppjieba.done ]; then
  pushd $MAIN_ROOT/tools && make cppjieba.done && popd
fi

ln -s $MAIN_ROOT/tools/cppjieba $MAIN_ROOT/core/ops/cppjieba || { echo "build ops: link jieba error" ; exit 1; }

# clean 

make clean &> /dev/null || exit 1

# compile custom ops under tensorflow/core/user_ops

if [ $target == 'delta' ];then
    make -j $(nproc)

    if [ ! -f ./x_ops.so ];then
        echo "No x_ops.so generated. Compiling ops failed!"
        exit 1
    fi

elif [ $target == 'deltann' ]; then
    ops_dir=$MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops/ops 
    if [ -L $ops_dir ] && [ -d $ops_dir ]; then
        unlink $MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops/ops
    fi
    ln -s $MAIN_ROOT/delta/layers/ops $MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops
    
    pushd $MAIN_ROOT/tools/tensorflow
   
    bazel build --verbose_failures -c opt //tensorflow/core/user_ops/ops:x_ops.so || { echo "compile custom ops error"; exit 1; }
    
    cp bazel-bin/tensorflow/core/user_ops/ops/*.so $MAIN_ROOT/dpl/lib/custom_ops
    cp $MAIN_ROOT/dpl/lib/custom_ops/x_ops.so $MAIN_ROOT/core/ops/

    popd
fi
