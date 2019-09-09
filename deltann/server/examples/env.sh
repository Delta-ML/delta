source env.sh

export DELTANN_MAIN=$MAIN_ROOT/../dpl/output/lib/deltann
export DELTANN_OPS=$MAIN_ROOT/../dpl/output/lib/custom_ops
export DELTANN_TENSORFLOW=$MAIN_ROOT/../dpl/output/lib/tensorflow

LD_LIBRARY_PATH=$DELTANN_MAIN:$DELTANN_OPS:$DELTANN_TENSORFLOW:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
