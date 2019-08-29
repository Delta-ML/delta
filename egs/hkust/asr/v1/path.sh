
if [ -z $MAIN_ROOT ];then
    source ../../../../env.sh
    echo "source env.sh"
fi

export LC_ALL=C
export PATH=$PATH:$PWD/utils/:$PWD
