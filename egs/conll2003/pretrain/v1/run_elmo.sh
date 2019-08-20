#!/bin/bash

start_stage=0
stop_stage=100
data=./data/
exp=./exp_elmo/
local=./local


if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # generate data from seqlabel
    cd ../../seq_label/v1
    ./run.sh
    cp -r data ../../pretrain/v1/
    cd ../../pretrain/v1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # preprocess data
    python local/preprocess_elmo_dataset.py $data/train.txt $data/dev.txt $data/test.txt
fi

if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    wget -P $data \
    https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    wget -P $data \
    https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
    git clone https://github.com/allenai/bilm-tf.git $local/bilm-tf
    mv $local/bilm-tf $local/bilm
fi

if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # generate elmo vocab && transfer standard model (read README.md)
    [ -d $exp ] || mkdir -p $exp
    python local/generate_elmo_vocab.py $data/ $exp/text_elmo_vocab.txt $exp/label_elmo_vocab.txt 0
    sed -n '2,$p' $exp/text_elmo_vocab.txt | cut -f 1 > $exp/elmo_vocab.txt
    python local/transfer_elmo_model.py $exp/elmo_vocab.txt $data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json \
    $data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 $data/token_embedding \
    $data/elmo_model_placeholder.ckpt
fi

