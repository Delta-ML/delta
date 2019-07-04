#!/bin/bash

set -x

if [[ $# != 1 ]];then
  echo "usage: $0 output/predict.txt"
  exit 1
fi

file=$1

tmp=$(mktemp /tmp/output.XXXXXXXXXX)

# extract filepath and postive_cnt
cat $1 | awk '{print $1, $NF}' > $tmp 

cat $tmp

python tools/order.py $tmp
