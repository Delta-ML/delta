#!/bin/bash

start_stage=0
stop_stage=100
data=./data
exp=./exp_bert
local=./local

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # generate data from seqlabel
    cd ../../seq_label/v1 || exit 1
    ./run.sh || exit 1
    cp -r data ../../pretrain/v1/ || exit 1
    cd ../../pretrain/v1 || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download model
    wget -P $data https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip || exit 1
    unzip -d $data/ $data/cased_L-24_H-1024_A-16.zip || exit 1
    git clone https://github.com/google-research/bert.git $local/bert || exit 1
    # overwrite files for tensorflow 2.0 compatibility
    cp $local/bert/modeling.py $local/bert/modeling.py.bak || exit 1
    cp $local/bert/tokenization.py $local/bert/tokenization.py.bak || exit 1
    cp $local/modeling.py $local/bert/ || exit 1
    cp $local/tokenization.py $local/bert/ || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # download && preprocess data
    python local/preprocess_bert_dataset.py $data/train.txt $data/dev.txt $data/test.txt \
    $data/cased_L-24_H-1024_A-16/vocab.txt|| exit 1
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # generate bert vocab
    [ -d $exp ] || mkdir -p $exp || exit 1
    python local/generate_bert_vocab.py $data/train.txt \
    $data/cased_L-24_H-1024_A-16/vocab.txt $exp/text_bert_vocab.txt $exp/text_bert_label.txt|| exit 1
    python local/transfer_bert_model.py $data/cased_L-24_H-1024_A-16 $data/bert_model_placeholder.ckpt|| exit 1
fi


