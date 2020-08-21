#!/bin/bash

if [ $# != 1 ];then
    echo "usage: $0 [delta|deltann]"
    exit 1
fi

# be careful:
# delta depend on tensorflow python package
# deltann not depend on tensorflow python package

target=$1

if [ -z $MAIN_ROOT ];then
    pushd ../../ && source env.sh && popd
    echo "source env.sh"
fi

set -e
set -u
set -o pipefail

# check tf compiler version
if [ $target == 'delta' ];then
    local_ver=`gcc --version | grep ^gcc | sed 's/^.* //g'`
    tf_ver=`python -c "import tensorflow as tf; print(tf.version.COMPILER_VERSION.split()[0]);"`
    if [  ${local_ver:0:1} -ne ${tf_ver:0:1} ];then
      echo "gcc version($local_ver) not compatiable with tf compile version($tf_ver)"
      exit -1
    fi
fi


# prepare dependency
echo "build ops: prepare dependency"

if [ -L $MAIN_ROOT/transform/tf_wrapper/ops/cppjieba ]; then
    unlink $MAIN_ROOT/transform/tf_wrapper/ops/cppjieba
fi

if ! [ -f $MAIN_ROOT/tools/cppjieba.done ]; then
  pushd $MAIN_ROOT/tools && make cppjieba.done && popd
fi

ln -sf $MAIN_ROOT/tools/cppjieba $MAIN_ROOT/transform/tf_wrapper/ops/cppjieba || { echo "build ops: link jieba error" ; exit 1; }

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
    ln -sf $MAIN_ROOT/core/ops $MAIN_ROOT/tools/tensorflow/tensorflow/core/user_ops

    python3 gen_build.py
    pushd $MAIN_ROOT/tools/tensorflow
    bazel build --verbose_failures -c opt //tensorflow/core/user_ops/ops:x_ops.so || { echo "compile custom ops error"; exit 1; }

    cp bazel-bin/tensorflow/core/user_ops/ops/*.so $MAIN_ROOT/dpl/lib/custom_ops
    cp $MAIN_ROOT/dpl/lib/custom_ops/x_ops.so $MAIN_ROOT/core/ops/

    popd
fi
