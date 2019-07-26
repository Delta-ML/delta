#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    python3 local/generate_mock_data.py data/train.txt data/dev.txt data/test.txt data/text_vocab.txt || exit 1
fi
