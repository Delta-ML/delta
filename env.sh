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

# need version
export TF_VER=2.3.0
export PY_VER=3.6

# root dir
export KALDI_ROOT=$MAIN_ROOT/tools/kaldi
export ESPNET_ROOT=$MAIN_ROOT/tools/espnet

# using env python, not kaldi tools python
OLD_PATH=$PATH

# Kaldi
[ -f $KALDI_ROOT/tools/env.sh  ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PATH:$MAIN_ROOT/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$MAIN_ROOT
if [ ! -f $KALDI_ROOT/tools/config/common_path.sh ]; then 
  echo >&2 "Note: Kaldi path $KALDI_ROOT/tools/config/common_path.sh is not found. Please ignore this message if you are an NLP user." 
else
  . $KALDI_ROOT/tools/config/common_path.sh
fi

# espnet
export PATH=$OLD_PATH:$PATH:$ESPNET_ROOT/utils:$ESPNET_ROOT/bin:
export OMP_NUM_THREADS=1

# delta utils
export PATH=$PATH:$MAIN_ROOT/utils

# pip bins
export PATH=$PATH:~/.local/bin

# delta
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT:$MAIN_ROOT/tools/espnet

# go
if [ -e $MAIN_ROOT/tools/go.env ];then
  source $MAIN_ROOT/tools/go.env
fi

# maybe used by deltann
# tensorflow lib path
#TF_LIB_PATH=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#export LD_LIBRARY_PATH=${TF_LIB_PATH}:${LD_LIBRARY_PATH}
