#!/bin/bash

start_stage=0
stop_stage=100
data=./data/
_url ="https://firebasestorage.googleapis.com/v0/b/mtl-sentence" \
       "-representations.appspot.com/o/data%2FQQP.zip?alt=media&" \
       "token=700c6acf-160d-4d89-81d1-de4191d02cb5"

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download quora_qp data path data/QQP/[train.tsv,test.tsv,dev.tsv]
    #to do
    python local/load_data.py _url $data ||exit 1

fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # generate_standard_format
    #label  sentence1   sentence2
    python local/generate_standard_format.py $data/QQP/original/quora_duplicate_questions.tsv $data/quora_stand.txt || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # split dev and train
    python utils/nlp/split_train_dev.py $data/quora_stand.txt $data/train_dev.txt $data/test.txt 0.1 || exit 1
    python utils/nlp/split_train_dev.py $data/train_dev.txt $data/train.txt $data/dev.txt 0.1 || exit 1
    rm $data/train_dev.txt
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # scale data
    python utils/nlp/scale.py $data/train.txt $data/train_small.txt 0.01 || exit 1
    python utils/nlp/scale.py $data/dev.txt $data/dev_small.txt 0.01 || exit 1
    python utils/nlp/scale.py $data/test.txt $data/test_small.txt 0.01 || exit 1
fi
