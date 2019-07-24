if [ -z $KALDI_ROOT ];then
    source ../../../env.sh
    echo "source env.sh"
fi

export PATH=$PWD/utils/:$PWD:$PATH
export LC_ALL=C
