if [ -z $MAIN_ROOT ];then
    pushd ../../../../ && source env.sh && pushd
    echo "source env.sh"
fi

export PATH=$PATH:$PWD/utils/:$PWD
