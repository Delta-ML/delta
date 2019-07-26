#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data https://github.com/howl-anderson/ATIS_dataset/raw/master/data/raw_data/ms-cntk-atis/atis.train.pkl || exit 1
    wget -P $data https://github.com/howl-anderson/ATIS_dataset/raw/master/data/raw_data/ms-cntk-atis/atis.test.pkl || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # generate data with standard format
    python3 local/summary_data.py $data/train.txt $data/test.txt || exit 1
fi
