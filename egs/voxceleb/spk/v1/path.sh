if [ -z $KALDI_ROOT ];then
    pushd ../../../.. && source env.sh && popd
    echo "source env.sh"
fi

export PATH=$PWD/utils/:$PWD:$PATH
export LC_ALL=C
