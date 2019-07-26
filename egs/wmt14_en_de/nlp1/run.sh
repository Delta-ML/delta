#!/bin/bash

start_stage=0
stop_stage=100
data=./data/
exp=./exp/
utils=../../../utils/nlp

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1;
    wget -P $data https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en || exit 1
    wget -P $data https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de || exit 1
    wget -P $data https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en || exit 1
    wget -P $data https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de || exit 1
    wget -P $data https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en || exit 1
    wget -P $data https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de || exit 1
    mv $data/newstest2013.en $data/val.en || exit 1
    mv $data/newstest2013.de $data/val.de || exit 1
    mv $data/newstest2014.en $data/test.en || exit 1
    mv $data/newstest2014.de $data/test.de || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # download vocab
    [ -d exp ] || mkdir -p exp || exit 1;
    wget -P $exp https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en || exit 1
    wget -P $exp https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de || exit 1
    python local/generate_stand_vocab.py $exp/vocab.50K.en $exp/en.vocab || exit 1
    python local/generate_stand_vocab.py $exp/vocab.50K.en $exp/en.vocab || exit 1
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # scale data
    python $utils/scale.py $data/train.en $data/train.en.small 0.05 || exit 1
    python $utils/scale.py $data/train.de $data/train.de.small 0.05 || exit 1
    python $utils/scale.py $data/val.en $data/val.en.small 0.05 || exit 1
    python $utils/scale.py $data/val.de $data/val.de.small 0.05 || exit 1
    python $utils/scale.py $data/test.en $data/test.en.small 0.05 || exit 1
    python $utils/scale.py $data/test.de $data/test.de.small 0.05 || exit 1
fi



