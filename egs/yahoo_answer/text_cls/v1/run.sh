#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1
    wget -P $data https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz || exit 1
    tar zxvf $data/yahoo_answers_csv.tgz  -C $data || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # generate data with standard format
    python local/generate_standard_format.py $data/yahoo_answers_csv/train.csv $data/train_all.txt || exit 1
    python local/generate_standard_format.py $data/yahoo_answers_csv/test.csv $data/test.txt || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # split dev and train
    python utils/nlp/split_train_dev.py $data/train_all.txt $data/train.txt $data/dev.txt 0.1 || exit 1
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # scale data
    python utils/nlp/scale.py $data/train.txt $data/train.small.txt 0.01 || exit 1
    python utils/nlp/scale.py $data/dev.txt $data/dev.small.txt 0.01 || exit 1
    python utils/nlp/scale.py $data/test.txt $data/test.small.txt 0.01 || exit 1
fi
