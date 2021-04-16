#!/bin/bash

start_stage=0
stop_stage=1
data=./data

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ" -O $data/cnn_stories.tgz && rm -rf /tmp/cookies.txt || exit 1    
    tar zxvf $data/cnn_stories.tgz  -C $data || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # split train, dev and test set
    git clone https://github.com/abisee/cnn-dailymail
    python local/make_datafiles.py $data $data || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # scale data
    python utils/scale.py $data/train.cnndm.src $data/train.small.cnndm.src 0.05 || exit 1
    python utils/scale.py $data/train.cnndm.tgt $data/train.small.cnndm.tgt 0.05 || exit 1
    python utils/scale.py $data/val.cnndm.src $data/val.small.cnndm.src 0.05 || exit 1
    python utils/scale.py $data/val.cnndm.tgt $data/val.small.cnndm.tgt 0.05 || exit 1
    python utils/scale.py $data/test.cnndm.src $data/test.small.cnndm.src 0.05 || exit 1
    python utils/scale.py $data/test.cnndm.tgt $data/test.small.cnndm.tgt 0.05 || exit 1
fi
