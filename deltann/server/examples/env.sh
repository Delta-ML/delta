# check if we are executing or sourcing.
if [[ "$0" == "$BASH_SOURCE" ]]
then
    echo "You must source this script rather than executing it."
    exit -1
fi

# don't use readlink because macOS won't support it.
if [[ "$BASH_SOURCE" == "/"* ]]
then
    export MAIN_ROOT=$(dirname $BASH_SOURCE)
else
    export MAIN_ROOT=$PWD/$(dirname $BASH_SOURCE)
fi

export DELTANN_MAIN=$MAIN_ROOT/../dpl/output/lib/deltann
export DELTANN_OPS=$MAIN_ROOT/../dpl/output/lib/custom_ops
export DELTANN_TENSORFLOW=$MAIN_ROOT/../dpl/output/lib/tensorflow

LD_LIBRARY_PATH=$DELTANN_MAIN:$DELTANN_OPS:$DELTANN_TENSORFLOW:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
