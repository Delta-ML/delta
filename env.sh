# root dir
export MAIN_ROOT=$PWD
export KALDI_ROOT=$MAIN_ROOT/tools/kaldi
export ESPNET_ROOT=$MAIN_ROOT/tools/espnet

# Kaldi
[ -f $KALDI_ROOT/tools/env.sh  ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
if [ ! -f $KALDI_ROOT/tools/config/common_path.sh ]; then 
  echo >&2 "Warning: Kaldi: The standard file $KALDI_ROOT/tools/config/common_path.sh is not present" 
else
  . $KALDI_ROOT/tools/config/common_path.sh
fi
export LC_ALL=C

#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

# espnet
export PATH=$ESPNET_ROOT/utils:$ESPNET_ROOT/bin:$MAIN_ROOT/utils:$PATH
export OMP_NUM_THREADS=1

# delta
export PYTHONPATH=$MAIN_ROOT:$MAIN_ROOT/tools/espnet
