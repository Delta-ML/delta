#!/bin/bash

set -e
set -x 

. path.sh

python3 -u $MAIN_ROOT/delta/main.py --config conf/asr-ctc.yml --cmd train_and_eval


