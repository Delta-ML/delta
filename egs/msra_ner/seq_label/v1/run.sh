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
    mv $data/train_data $data/train.in.txt || exit 1
    mv $data/test_data $data/dev.in.txt || exit 1
    mv $data/test1.txt $data/test.in.txt || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # change data format
    [ -d $data ] || mkdir -p $data || exit 1;
    python3 local/change_data_format.py data/train.in.txt data/dev.in.txt data/test.in.txt || exit 1
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # change name
    mv $data/train.out.txt $data/train.txt || exit 1
    mv $data/dev.out.txt $data/dev.txt || exit 1
    mv $data/test.out.txt $data/test.txt || exit 1
fi

if [ ${start_stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # scale data
    python3 utils/nlp/scale.py $data/train.txt $data/train.small.txt 0.05 || exit 1
    python3 utils/nlp/scale.py $data/dev.txt $data/dev.small.txt 0.05 || exit 1
    python3 utils/nlp/scale.py $data/test.txt $data/test.small.txt 0.05 || exit 1
fi
