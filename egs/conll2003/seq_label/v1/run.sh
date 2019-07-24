#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/train.txt || exit 1
    wget -P $data https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/dev.txt || exit 1
    wget -P $data https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/test.txt || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # change name
    mv $data/train.txt $data/train.in.txt || exit 1
    mv $data/dev.txt $data/dev.in.txt || exit 1
    mv $data/test.txt $data/test.in.txt || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # change data format
    [ -d $data ] || mkdir -p $data || exit 1;
    python local/change_data_format.py data/train.in.txt data/dev.in.txt data/test.in.txt || exit 1
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # change name
    mv $data/train.out.txt $data/train.txt || exit 1
    mv $data/dev.out.txt $data/dev.txt || exit 1
    mv $data/test.out.txt $data/test.txt || exit 1
fi

if [ ${start_stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # scale data
    python utils/nlp/scale.py $data/train.txt $data/train.small.txt 0.05 || exit 1
    python utils/nlp/scale.py $data/dev.txt $data/dev.small.txt 0.05 || exit 1
    python utils/nlp/scale.py $data/test.txt $data/test.small.txt 0.05 || exit 1
fi
