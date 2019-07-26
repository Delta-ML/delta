#!/bin/bash

set -x 

. path.sh

CUDA_VISIBLE_DEVICES=2,3 python3 -u $MAIN_ROOT/delta/main.py --config conf/asr-ctc.yml --cmd train


