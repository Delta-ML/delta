#!/bin/bash


if [[ $# != 2 ]];then
  echo "usage: $0 graph_file binary"
  exit 1
fi

file=$1
binary=$2

python3 tools/converter_grpah.py --graph_file=$file --binary=$binary
