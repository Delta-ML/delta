#!/bin/bash

start_stage=0
stop_stage=100
data=./data/
url=https://nlp.stanford.edu/projects/snli/snli_1.0.zip

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data $url || exit 1
    unzip $data/snli_1.0.zip  -d $data || exit 1

fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # generate_standard_format
    #label  sentence1   sentence2
    python3 local/generate_standard_format.py $data/snli_1.0/snli_1.0_dev.jsonl $data/dev.txt || exit 1
    python3 local/generate_standard_format.py $data/snli_1.0/snli_1.0_test.jsonl $data/test.txt || exit 1
    python3 local/generate_standard_format.py $data/snli_1.0/snli_1.0_train.jsonl $data/train.txt || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # scale data
    python3 utils/nlp/scale.py $data/train.txt $data/train_small.txt 0.05 || exit 1
    python3 utils/nlp/scale.py $data/dev.txt $data/dev_small.txt 0.05 || exit 1
    python3 utils/nlp/scale.py $data/test.txt $data/test_small.txt 0.05 || exit 1
fi
