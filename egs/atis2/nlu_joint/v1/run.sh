#!/bin/bash

start_stage=0
stop_stage=100
data=./data/

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # download data
    [ -d $data ] || mkdir -p $data || exit 1
    git clone https://github.com/yvchen/JointSLU.git JointSLU || exit 1
    mv JointSLU/data origin_data && rm -r -f JointSLU || exit 1
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # generate data with standard format
    python3 local/generate_standard_format.py origin_data/atis-2.train.w-intent.iob $data/train.txt || exit 1
    python3 local/generate_standard_format.py origin_data/atis-2.dev.w-intent.iob $data/dev.txt || exit 1
    python3 local/generate_standard_format.py origin_data/atis.test.w-intent.iob $data/test.txt || exit 1
fi
