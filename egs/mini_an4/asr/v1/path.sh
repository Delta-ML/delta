if [ -z $MAIN_ROOT ];then
    source ../../../../env.sh
    echo "source env.sh"
fi

export LC_ALL=C
# https://github.com/espnet/espnet/pull/1090
export PYTHONIOENCODING=UTF-8

export PATH=$PATH:$PWD/utils/:$PWD
