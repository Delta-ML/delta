
. ./path.sh

set -e

start_stage=0
end_stage=6

iemocap_root=/export/corpus/iemocap # dataset root dir

. utils/parse_options.sh # e.g. this parses the --stage option if supplied.

if [[ $start_stage -le 0 && ! -d "data" ]]; then
    echo "mkdir data"
    mkdir data
fi

#1. link `iemocap dir` to `sessions`
if [[ $start_stage -le 1 && ! -L "data/sessions" ]]; then
    echo "link iemocap"
    ln -s $iemocap_root data/sessions
fi

#2. collect data
if [[ $start_stage -le 2 && ! -f "data/data_collected.pickle" ]]; then
    echo "collect data"
    python3 -u local/python/mocap_data_collect.py
fi

#3. to save `wav`, `text`, `label` to  `dmpy dir
if [[ $start_stage -le 3 && ! -d "./data/dump" ]]; then
    echo "dump"
    mkdir ./data/dump
    python3 local/python/dump_data_from_pickle.py
fi

config_file=conf/emo-keras-resnet50.yml
echo "Using config $config_file"

#4. make fbank
if [[ $start_stage -le 4 ]];then
    echo "make fbank"
    python3 -u $MAIN_ROOT/delta/main.py --cmd gen_feat --config $config_file
fi

#5. make cmvn
if [[ $start_stage -le 5 ]]; then
    echo "make cmvn"
    python3 -u $MAIN_ROOT/delta/main.py --cmd gen_cmvn --config $config_file
fi

#6. train
if [ $start_stage -le 6 ] && [ $end_stage -ge 6 ]; then
    echo "tain and eval"
    python3 -u $MAIN_ROOT/delta/main.py --cmd train_and_eval --config $config_file
fi
