#!/bin/bash


if [[ $# != 1 ]];then
  echo "usage: $0 dataset_path"
  exit 1
fi

data_path=$1

find $data_path -name '*.TextGrid' > textgrid.list


python3 generate_segment_from_textgrid.py textgrid.list textgrid.segments

cp textgrid.segments $data_path 
