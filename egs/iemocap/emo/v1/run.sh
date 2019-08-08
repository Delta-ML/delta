#!/bin/bash

. ./path.sh

set -e

start_stage=0
end_stage=100

iemocap_root=/export/corpus/iemocap # dataset root dir

. utils/parse_options.sh # e.g. this parses the --stage option if supplied.

if [[ $start_stage -le 0 && $end_stage -gt 0 && ! -d "data" ]]; then
    echo "mkdir data"
    mkdir -p data
fi

#1. collect data
if [[ $start_stage -le 1 && $end_stage -gt 1 && ! -f "data/data_collected.pickle" ]]; then
    echo "collect data"
    python3 -u local/python/mocap_data_collect.py $iemocap_root
fi

#2. to save `wav`, `text`, `label` to  `dmpy dir
if [[ $start_stage -le 2 && $end_stage -gt 2 && ! -d "./data/dump" ]]; then
    echo "dump"
    mkdir -p data/dump
    python3 local/python/dump_data_from_pickle.py
fi

config_file=conf/emo-keras-blstm.yml
echo "Using config $config_file"

#3. make fbank
if [[ $start_stage -le 3 && $end_stage -gt 3 ]];then
    echo "make fbank"
    python3 -u $MAIN_ROOT/delta/main.py --cmd gen_feat --config $config_file
fi

#4. make cmvn
if [[ $start_stage -le 4 && $end_stage -gt 4 ]]; then
    echo "make cmvn"
    python3 -u $MAIN_ROOT/delta/main.py --cmd gen_cmvn --config $config_file
fi

#5. train and eval
if [[ $start_stage -le 5 && $end_stage -gt 5 ]]; then
    echo "tain and eval"
    python3 -u $MAIN_ROOT/delta/main.py --cmd train_and_eval --config $config_file
fi
