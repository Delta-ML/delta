#!/bin/bash

if [[ $# != 1 ]]; then
  echo "usage: $0 output/predict_score.txt"
  exit 1
fi

# score_file : `filename, .., score`
python tools/roc_curve.py $1
