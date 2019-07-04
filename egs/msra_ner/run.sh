#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data https://raw.githubusercontent.com/Determined22/zh-NER-TF/master/data_path/train_data || exit 1
    wget -P $data https://raw.githubusercontent.com/Determined22/zh-NER-TF/master/data_path/test_data || exit 1
    wget -P $data https://raw.githubusercontent.com/Super-Louis/keras-crf-ner/master/data/original/test1.txt || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # change name
    mv $data/train_data $data/train.txt || exit 1
    mv $data/test_data $data/dev.txt || exit 1
    mv $data/test1.txt $data/test.txt || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # scale data
    python utils/scale.py $data/train.txt $data/train.small.txt 0.05 || exit 1
    python utils/scale.py $data/dev.txt $data/dev.small.txt 0.05 || exit 1
    python utils/scale.py $data/test.txt $data/test.small.txt 0.05 || exit 1
fi
