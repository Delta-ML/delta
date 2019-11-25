#!/bin/bash

stage=-1
stop_stage=100
config_file=asr-ctc.yml

source path.sh
source $MAIN_ROOT/utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export CUDA_VISIBLE_DEVICES=''

echo "Running from stage $stage ..."

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  echo "Prepare data..."
  bash run.sh --stage -1 --stop_stage 2
  echo "Prepare data done."
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Train and Eval..."
  python3 -u $MAIN_ROOT/delta/main.py --config conf/$config_file --cmd train_and_eval
  echo "Train and Eval Done."
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Train..."
  python3 -u $MAIN_ROOT/delta/main.py --config conf/$config_file --cmd train
  echo "Train Done."
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Eval..."
  python3 -u $MAIN_ROOT/delta/main.py --config conf/$config_file --cmd eval
  echo "Eval Done."
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Infer..."
  python3 -u $MAIN_ROOT/delta/main.py --config conf/$config_file --cmd infer
  echo "Infer Done."
fi
