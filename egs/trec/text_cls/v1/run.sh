#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data https://raw.githubusercontent.com/thtrieu/qclass_dl/master/data/train || exit 1
    wget -P $data https://raw.githubusercontent.com/thtrieu/qclass_dl/master/data/test || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # change data format
    [ -d $data ] || mkdir -p $data || exit 1;
    python3 local/change_data_format.py $data/train $data/train.txt|| exit 1
    python3 local/change_data_format.py $data/test $data/test.txt || exit 1
fi
