#!/bin/bash

start_stage=0
stop_stage=100
data=./data/
url='http://10.84.138.11:8015/kc.tar.gz'

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data $url || exit 1
    tar zxvf $data/kc.tar.gz  -C $data || exit 1
fi

